"""
mneme.pass_plugins

User LLVM pass plugins for replay.

A :class:`PassPlugins` value names plugin shared libraries to load into the
Proteus JIT and the pipeline fragments to run before and after each
experiment's pipeline. The fragments stay fixed while a tuning search varies
``ExperimentConfiguration.passes``.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_BARE_OPT_LEVELS = {"O0", "O1", "O2", "O3", "Os", "Oz"}


@dataclass(frozen=True)
class PassPlugins:
    """
    Pass plugin libraries and the pipeline fragments that surround every
    replay pipeline.

    Attributes
    ----------
    libraries : tuple[str, ...]
        Plugin shared libraries. Each is loaded before optimization so that its
        pass names parse in ``prepend``, ``append`` and the experiment pipeline.
    prepend : str, optional
        Pipeline fragment run before the experiment pipeline, e.g. ``"my-pass"``.
    append : str, optional
        Pipeline fragment run after the experiment pipeline.
    """

    libraries: Tuple[str, ...] = ()
    prepend: Optional[str] = None
    append: Optional[str] = None

    def __post_init__(self):
        libraries = tuple(os.path.abspath(path) for path in self.libraries)
        missing = [path for path in libraries if not os.path.isfile(path)]
        if missing:
            raise ValueError(f"Pass plugin library not found: {', '.join(missing)}")
        object.__setattr__(self, "libraries", libraries)

    @classmethod
    def from_options(
        cls,
        pass_plugins: Optional[Sequence[str]] = None,
        prepend_passes: Optional[str] = None,
        append_passes: Optional[str] = None,
    ) -> "PassPlugins":
        """Build from the ``--pass-plugin``/``--prepend-passes``/``--append-passes`` options."""
        return cls(tuple(pass_plugins or ()), prepend_passes or None, append_passes or None)

    def wrap(self, passes: str) -> str:
        """Return the textual pipeline that runs ``passes`` between the fragments."""
        return compose_pipeline(self.prepend, passes, self.append)


def compose_pipeline(*fragments: Optional[str]) -> str:
    """
    Join pipeline fragments into one textual LLVM pipeline.

    Proteus treats a bare level such as ``O3`` as an optimization level rather
    than a pipeline, so it cannot be joined with other passes. Bare levels are
    therefore spelled as ``default<O3>``. Empty fragments are dropped.
    """
    return ",".join(
        f"default<{fragment}>" if fragment in _BARE_OPT_LEVELS else fragment
        for fragment in fragments
        if fragment
    )


def add_pass_plugin_args(parser) -> None:
    """Add the pass plugin options shared by ``mneme replay`` and ``mneme tune``."""
    parser.add_argument(
        "--pass-plugin",
        dest="pass_plugins",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "LLVM pass plugin shared library to load into the JIT, making its "
            "passes available to the pipeline. May be repeated."
        ),
    )
    parser.add_argument(
        "--prepend-passes",
        dest="prepend_passes",
        default=None,
        metavar="PIPELINE",
        help="Pass pipeline to run before the kernel's pipeline, e.g. a pass from --pass-plugin.",
    )
    parser.add_argument(
        "--append-passes",
        dest="append_passes",
        default=None,
        metavar="PIPELINE",
        help="Pass pipeline to run after the kernel's pipeline.",
    )
