"""HDF5 file access mode enumeration."""

from __future__ import annotations

from enum import Enum


class HDF5AccessMode(Enum):
    """Access mode under which an HDF5 file is opened.

    Only the two modes that matter to :class:`HDF5Container` are modeled:
    a strictly read-only mode and an append mode that also permits
    creating new groups and datasets. HDF5's other native modes
    (truncate-create, exclusive-create) are handled separately by
    :meth:`HDF5Container.from_path`.
    """

    READ_ONLY = "r"
    APPEND = "a"

    @property
    def h5py_mode(self) -> str:
        """Return the mode string accepted by ``h5py.File``.

        Returns
        -------
        str
            The h5py-compatible file mode.
        """
        return self.value

    @property
    def is_writable(self) -> bool:
        """Return whether this access mode permits mutating the file.

        Returns
        -------
        bool
            ``True`` when groups and datasets may be created or modified.
        """
        match self:
            case HDF5AccessMode.READ_ONLY:
                return False
            case HDF5AccessMode.APPEND:
                return True
