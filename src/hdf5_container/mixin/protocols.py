from __future__ import annotations

import threading
from typing import Protocol

import h5py

from ..options import HDF5AccessMode, HDF5FileOptions
from ..utils.counter import FlushCounter


class HDF5ContainerProtocol(Protocol):
    """Structural contract shared by container mixins.

    Mixins rely on these attributes and methods to provide behavior while
    staying decoupled from the concrete container class.
    """

    data: h5py.File | h5py.Group
    flush_interval: int
    counter: FlushCounter
    lock: threading.RLock
    path: str | None
    access_mode: HDF5AccessMode | None
    file_options: HDF5FileOptions
    is_page_buffering_enabled: bool

    def __init__(
        self,
        data: h5py.File | h5py.Group,
        flush_interval: int = 100,
        counter: FlushCounter = ...,
        lock: threading.RLock = ...,
        path: str | None = None,
        access_mode: HDF5AccessMode | None = None,
        file_options: HDF5FileOptions = ...,
        is_page_buffering_enabled: bool = False,
    ) -> None:
        """Construct a compatible container instance."""

    def flush(self) -> None:
        """Flush pending data to disk."""

    def close(self) -> None:
        """Close the underlying HDF5 file resources."""
