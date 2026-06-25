from __future__ import annotations

import os
from typing import TypeVar

import h5py

from .protocols import HDF5ContainerProtocol

TContainer = TypeVar("TContainer", bound=HDF5ContainerProtocol)


class IOMixin:
    """I/O behavior for file-backed HDF5 containers."""

    @classmethod
    def from_path(
        cls: type[TContainer],
        path: str,
        flush_interval: int = 100,
    ) -> TContainer:
        """Create a container from an HDF5 file path.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to open/create.
        flush_interval : int, optional
            Number of write operations between automatic flushes.

        Returns
        -------
        TContainer
            A container instance backed by the target HDF5 file.
        """
        is_exist_file = os.path.exists(path)
        data = h5py.File(path, "a")
        if not is_exist_file:
            data.flush()
        return cls(  # type: ignore[call-arg]
            data=data,
            flush_interval=flush_interval,
        )

    def flush(self: HDF5ContainerProtocol) -> None:
        """Force buffered HDF5 changes to be written to disk.

        Notes
        -----
        If the container wraps a group, the parent file is flushed.
        This method is thread-safe via internal locking.
        """
        with self.lock:
            if isinstance(self.data, h5py.File):
                self.data.flush()
            else:
                self.data.file.flush()

    def close(self: HDF5ContainerProtocol) -> None:
        """Flush pending updates and close the underlying HDF5 file.

        Notes
        -----
        This method is safe for both file-backed and group-backed containers.
        This method is thread-safe via internal locking.
        """
        with self.lock:
            if isinstance(self.data, h5py.File):
                self.data.flush()
            else:
                self.data.file.flush()
            if isinstance(self.data, h5py.File):
                self.data.close()
            else:
                self.data.file.close()
