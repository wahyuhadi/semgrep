import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Sequence
from typing import Set

import attr
from wcmatch import glob as wcglob

from semgrep.config_resolver import resolve_targets
from semgrep.error import FilesNotFoundError
from semgrep.output import OutputHandler
from semgrep.semgrep_types import Language
from semgrep.target_manager_extensions import accept_path_for_lang
from semgrep.target_manager_extensions import ALL_EXTENSIONS
from semgrep.target_manager_extensions import lang_to_exts
from semgrep.util import partition_set
from semgrep.util import sub_check_output
from semgrep.verbose_logging import getLogger

logger = getLogger(__name__)


@contextlib.contextmanager
def optional_stdin_target(target: Sequence[str]) -> Iterator[Sequence[str]]:
    """
    Read target input from stdin if "-" is specified
    """
    if target == ["-"]:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as fd:
                fd.write(sys.stdin.buffer.read())
                fname = fd.name
            yield [fname]
        finally:
            os.remove(fname)
    else:
        yield target


# Target files obtained from the command line (explicit) and discovered
# by scanning folders specified on the command line (filterable).
# The latter are subject to further filtering by semgrep-core, which is
# why we keep them separate and we let semgrep-core know which are which.
class TargetFiles(NamedTuple):
    explicit: Set[Path]
    filterable: Set[Path]


@attr.s(auto_attribs=True)
class TargetManager:
    """
    Handles all file include/exclude logic for semgrep

    If respect_git_ignore is true then will only consider files that are
    tracked or (untracked but not ignored) by git

    If skip_unknown_extensions is False then targets with extensions that are
    not understood by semgrep will always be returned by get_files. Else will discard
    targets with unknown extensions
    """

    includes: Sequence[str]
    excludes: Sequence[str]
    targets: Sequence[str]  # explicit target files or directories
    respect_git_ignore: bool
    output_handler: OutputHandler
    skip_unknown_extensions: bool

    # For each language, a pair (explicit target files, filterable target files)
    _filtered_targets: Dict[Language, TargetFiles] = attr.ib(factory=dict)

    @staticmethod
    def resolve_targets(targets: Sequence[str]) -> Set[Path]:
        """
        Return list of Path objects appropriately resolving relative paths
        (relative to cwd) if necessary
        """
        return set(resolve_targets(targets))

    @staticmethod
    def _is_valid_file_or_dir(path: Path) -> bool:
        """Check this is a valid file or directory for semgrep scanning."""
        return os.access(path, os.R_OK) and not path.is_symlink()

    @staticmethod
    def _is_valid_file(path: Path) -> bool:
        """Check if file is a readable regular file.

        This eliminates files that should never be semgrep targets. Among
        others, this takes care of excluding symbolic links (because we don't
        want to scan the target twice), directories (which may be returned by
        globbing or by 'git ls-files' e.g. submodules), and files missing
        the read permission.
        """
        return TargetManager._is_valid_file_or_dir(path) and path.is_file()

    @staticmethod
    def _filter_valid_files(paths: Set[Path]) -> Set[Path]:
        """Keep only readable regular files"""
        return set(path for path in paths if TargetManager._is_valid_file(path))

    @staticmethod
    def _list_files_in_dir(curr_dir: Path) -> Set[Path]:
        """Return set of all files in curr_dir."""
        return set(p for p in curr_dir.rglob("*") if TargetManager._is_valid_file(p))

    @staticmethod
    def _parse_output(output: str, curr_dir: Path) -> Set[Path]:
        """
        Convert a newline delimited list of files to a set of path objects
        prepends curr_dir to all paths in said list

        If list is empty then returns an empty set
        """
        files: Set[Path] = set()
        if output:
            files = set(
                p
                for p in (Path(curr_dir) / elem for elem in output.strip().split("\n"))
                if TargetManager._is_valid_file(p)
            )
        return files

    @staticmethod
    def _expand_dir(curr_dir: Path, respect_git_ignore: bool) -> Set[Path]:
        """Recursively go through a directory and return list of all files."""

        expanded: Set[Path] = set()

        if respect_git_ignore:
            try:
                # Tracked files
                tracked_output = sub_check_output(
                    ["git", "ls-files"],
                    cwd=curr_dir.resolve(),
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )

                # Untracked but not ignored files
                untracked_output = sub_check_output(
                    [
                        "git",
                        "ls-files",
                        "--other",
                        "--exclude-standard",
                    ],
                    cwd=curr_dir.resolve(),
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )

                deleted_output = sub_check_output(
                    ["git", "ls-files", "--deleted"],
                    cwd=curr_dir.resolve(),
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.verbose(
                    f"Unable to ignore files ignored by git ({curr_dir} is not a git directory or git is not installed). Running on all files instead..."
                )
                # Not a git directory or git not installed. Fallback to using rglob
                files = TargetManager._list_files_in_dir(curr_dir)
                expanded = expanded.union(files)
            else:
                tracked = TargetManager._parse_output(tracked_output, curr_dir)
                untracked_unignored = TargetManager._parse_output(
                    untracked_output, curr_dir
                )
                deleted = TargetManager._parse_output(deleted_output, curr_dir)
                expanded = expanded.union(tracked)
                expanded = expanded.union(untracked_unignored)
                expanded = expanded.difference(deleted)

        else:
            files = TargetManager._list_files_in_dir(curr_dir)
            expanded = expanded.union(files)

        return TargetManager._filter_valid_files(expanded)

    @staticmethod
    def expand_targets(
        targets: Collection[Path], respect_git_ignore: bool
    ) -> Set[Path]:
        """Explore all directories."""
        expanded: Set[Path] = set()
        for target in targets:
            if not TargetManager._is_valid_file_or_dir(target):
                continue

            if target.is_dir():
                expanded.update(TargetManager._expand_dir(target, respect_git_ignore))
            else:
                expanded.add(target)

        return expanded

    @staticmethod
    def preprocess_path_patterns(patterns: Sequence[str]) -> List[str]:
        """Convert semgrep's path include/exclude patterns to wcmatch's glob patterns.

        In semgrep, pattern "foo/bar" should match paths "x/foo/bar", "foo/bar/x", and
        "x/foo/bar/x". It implicitly matches zero or more directories at the beginning and the end
        of the pattern. In contrast, we have to explicitly specify the globstar (**) patterns in
        wcmatch. This function will converts a pattern "foo/bar" into "**/foo/bar" and
        "**/foo/bar/**". We need the pattern without the trailing "/**" because "foo/bar.py/**"
        won't match "foo/bar.py".
        """
        result = []
        for pattern in patterns:
            result.append("**/" + pattern)
            result.append("**/" + pattern + "/**")
        return result

    @staticmethod
    def _filter_language(paths: Set[Path], lang: Language) -> Set[Path]:
        """Return subset of all the files that may be in the language.

        This should catch all the files that might be in the language.
        Semgrep-core will make a more thorough filtering.
        """
        return set(p for p in paths if accept_path_for_lang(p, lang))

    @staticmethod
    def filter_includes(paths: Set[Path], includes: Sequence[str]) -> Set[Path]:
        """Return all elements in paths that match any includes pattern.

        If includes is empty, returns paths unchanged
        """
        if not includes:
            return paths
        includes = TargetManager.preprocess_path_patterns(includes)
        return set(
            wcglob.globfilter(paths, includes, flags=wcglob.GLOBSTAR | wcglob.DOTGLOB)
        )

    @staticmethod
    def filter_excludes(paths: Set[Path], excludes: Sequence[str]) -> Set[Path]:
        """Returns all elements in arr that do not match any excludes pattern.

        If excludes is empty, returns arr unchanged
        """
        if not excludes:
            return paths

        excludes = TargetManager.preprocess_path_patterns(excludes)
        return paths - set(
            wcglob.globfilter(paths, excludes, flags=wcglob.GLOBSTAR | wcglob.DOTGLOB)
        )

    def filtered_files(self, lang: Language) -> TargetFiles:
        """Return files that should be analyzed for a language.

        This is a lazy computation. Scanning the file system is done only on
        the first call of this method.

        Target directories specified on the command line (or during object
        creation) are used as scanning roots to discover target files.
        Such discovered files are filtered out based on file extensions
        required by the language or other generic criteria.
        User-specified glob patterns are used to include or exclude certain
        paths or file names in addition to this.

        Files that are not directories are considered explicit targets
        and by default are not filtered out by any mechanism.
        """
        if lang in self._filtered_targets:
            return self._filtered_targets[lang]

        targets = self.resolve_targets(self.targets)

        files, directories = partition_set(lambda p: not p.is_dir(), targets)

        # Error on non-existent files
        explicit_files, nonexistent_files = partition_set(lambda p: p.is_file(), files)
        if nonexistent_files:
            self.output_handler.handle_semgrep_error(
                FilesNotFoundError(tuple(nonexistent_files))
            )

        # Scan file system.
        all_targets = self.expand_targets(directories, self.respect_git_ignore)

        # Filter based on file type and custom glob patterns.
        lang_targets = self._filter_language(all_targets, lang)
        included_targets = self.filter_includes(all_targets, self.includes)
        excluded_targets = self.filter_excludes(all_targets, [*self.excludes, ".git"])
        targets = lang_targets.union(included_targets).difference(excluded_targets)

        # Avoid duplicates (e.g. foo/bar can be both an explicit file and
        # discovered in folder foo/)
        targets = targets.difference(explicit_files)

        # Remove explicit targets with *known* extensions.
        # This violates "process all target files explicitly passed on the
        # command line".
        #
        # For now, is the best solution we have for dealing with a rule
        # that works for multiple languages. We exclude the explicit target
        # if it has a well-known extension that's not for the requested
        # language.
        # See https://github.com/returntocorp/semgrep/issues/966
        #
        # A better solution would be to not filter a target against a single
        # language. Instead, the list of allowed languages would stay as a list,
        # and we would pass (target, [lang1, lang2]) to semgrep-core.
        # semgrep-core would then try one language and then the other
        # if needed, which would avoid duplicate matches and would avoid
        # reporting a parsing error if parsing was successful with one language.
        #
        explicit_files_without_standard_extension = set(
            f
            for f in explicit_files
            if not any(f.match(f"*{ext}") for ext in ALL_EXTENSIONS)
        )

        explicit_files_with_expected_extension = set(
            f
            for f in explicit_files
            if any(f.match(f"*{ext}") for ext in lang_to_exts(lang))
        )

        # Optionally ignore explicit files with incorrect extensions for the
        # language (CLI option --skip-unknown-extensions).
        if self.skip_unknown_extensions:
            explicit_files = explicit_files_with_expected_extension
        else:  # default
            explicit_files = explicit_files_with_expected_extension.union(
                explicit_files_without_standard_extension
            )

        self._filtered_targets[lang] = TargetFiles(
            explicit=explicit_files, filterable=targets
        )
        return self._filtered_targets[lang]

    def get_files(
        self, lang: Language, extra_includes: List[str], extra_excludes: List[str]
    ) -> TargetFiles:
        """Return target files with extra glob patterns to include or exclude.

        This is meant for adding or removing target files from the default
        set on a rule-specific basis.
        """
        explicit_targets, filterable_targets = self.filtered_files(lang)
        filterable_targets = self.filter_includes(filterable_targets, extra_includes)
        filterable_targets = self.filter_excludes(filterable_targets, extra_excludes)
        return TargetFiles(explicit=explicit_targets, filterable=filterable_targets)
