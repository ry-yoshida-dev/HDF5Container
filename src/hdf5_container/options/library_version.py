"""HDF5 library version bound enumeration."""

from __future__ import annotations

from enum import Enum


class HDF5LibraryVersion(Enum):
    """Library format version bound passed to h5py's ``libver`` argument.

    Each member names a lower bound on the on-disk object format that HDF5
    is allowed to use. A newer bound unlocks features such as paged
    allocation and single-writer/multiple-reader support at the cost of
    losing compatibility with older HDF5 library versions.
    """

    EARLIEST = "earliest"
    V108 = "v108"
    V110 = "v110"
    V112 = "v112"
    LATEST = "latest"

    @property
    def bound(self) -> str:
        """Return the string accepted by h5py's ``libver`` parameter.

        Returns
        -------
        str
            The h5py-compatible library version bound.
        """
        return self.value
