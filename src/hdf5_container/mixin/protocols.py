from __future__ import annotations

import threading
from typing import Any, Protocol

import h5py


class HDF5ContainerProtocol(Protocol):
    """Structural contract shared by container mixins.

    Mixins rely on these attributes and methods to provide behavior while
    staying decoupled from the concrete container class.
    """

    data: h5py.File | h5py.Group
    flush_interval: int
    counter: Any
    lock: threading.RLock

    def __init__(
        self,
        data: h5py.File | h5py.Group,
        flush_interval: int = 100,
        counter: Any = ...,
        lock: threading.RLock = ...,
    ) -> None:
        """Construct a compatible container instance."""

    def flush(self) -> None:
        """Flush pending data to disk."""

    def close(self) -> None:
        """Close the underlying HDF5 file resources."""
