"""HDF5 file space management strategy enumeration."""

from __future__ import annotations

from enum import Enum


class HDF5FileSpaceStrategy(Enum):
    """Strategy HDF5 uses to track and reuse free space within a file.

    The value of each member is the string accepted by h5py's
    ``fs_strategy`` file-creation argument. ``PAGE`` groups metadata and
    small raw-data allocations into fixed-size pages, which is what keeps
    per-object overhead low for files holding many small datasets; it can
    only be selected when the file is created, never when reopening an
    existing file.
    """

    FSM_AGGR = "fsm"
    PAGE = "page"
    AGGR = "aggregate"
    NONE = "none"
