"""
mneme.pass_plugins

User LLVM pass plugins for replay.

Plugin shared libraries are loaded into the Proteus JIT so that their passes
can be named in ``ExperimentConfiguration.passes`` like any built-in pass.
"""

import os
from typing import Optional, Sequence, Tuple


def resolve_pass_plugins(libraries: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    Return the absolute paths of the pass plugin ``libraries``.

    Raises
    ------
    ValueError
        If any library does not exist.
    """
    resolved = tuple(os.path.abspath(path) for path in libraries or ())
    missing = [path for path in resolved if not os.path.isfile(path)]
    if missing:
        raise ValueError(f"Pass plugin library not found: {', '.join(missing)}")
    return resolved


def add_pass_plugin_args(parser) -> None:
    """Add the pass plugin option shared by ``mneme replay`` and ``mneme tune``."""
    parser.add_argument(
        "--pass-plugin",
        dest="pass_plugins",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "LLVM pass plugin shared library to load into the JIT, so that its "
            "passes can be named in pass pipelines. May be repeated."
        ),
    )
