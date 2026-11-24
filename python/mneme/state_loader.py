"""Size of the reusable pinned buffer used to load replay snapshots."""

import argparse
import re
import sys

# Experimentally found to be ideal on Mi300A loading from Lustre
# other systems and software stacks may be different
DEFAULT_BUFFER_SIZE = 32 << 20


def buffer_bytes(value):
    """ A nice reader for human readable buffer sizes to make the CLI a bit easier to use
    """
    match = re.fullmatch(r"([0-9]+)(B|KiB|MiB|GiB)?", str(value), re.IGNORECASE)
    if not match:
        raise ValueError("buffer size must be a byte count or an integer with KiB, MiB or GiB")

    scale = {"b": 1, "kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30}
    size = int(match[1]) * scale[(match[2] or "B").lower()]
    if not 0 < size <= sys.maxsize:
        raise ValueError("buffer size must be positive and fit in a signed pointer")
    return size


def add_buffer_size_arg(parser):
    parser.add_argument(
        "--buffer-size", type=buffer_bytes, default=argparse.SUPPRESS,
        help="Size of the snapshot staging buffer, e.g. 4MiB (default: 32MiB)",
    )
