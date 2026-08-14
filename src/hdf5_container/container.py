from __future__ import annotations
import logging
import threading
import h5py
import numpy as np
from typing import Any, Iterator
from dataclasses import dataclass, field

from .mixin import IOMixin, SpecialMethodsMixin
from .options import HDF5AccessMode, HDF5FileOptions
from .utils.counter import FlushCounter

logger = logging.getLogger(__name__)


@dataclass
class HDF5Container(IOMixin, SpecialMethodsMixin):
    """Container for HDF5 operations with automatic flushing.

    This class provides a convenient interface for storing and retrieving data
    from HDF5 files with hierarchical group structure support. Thread-safe
    operations are enforced through a reentrant lock shared across all
    subcontainers.

    Parameters
    ----------
    data : h5py.File | h5py.Group
        The HDF5 file or group to operate on.
    flush_interval : int, optional
        Number of operations before automatic flush, by default 100.
    counter : FlushCounter, optional
        Shared flush counter for periodic flushing.
    lock : threading.RLock, optional
        Shared reentrant lock for thread-safe access. If None, creates a new lock.
    path : str | None, optional
        Filesystem path this container was opened from via
        :meth:`from_path`, by default None. ``None`` for containers built
        directly from an ``h5py.File``/``h5py.Group`` without going
        through :meth:`from_path`.
    access_mode : HDF5AccessMode | None, optional
        Access mode the underlying file was opened with, by default None.
        ``None`` means no read-only enforcement is applied.
    file_options : HDF5FileOptions, optional
        Tuning options the underlying file was created/opened with.
    is_page_buffering_enabled : bool, optional
        Whether the underlying file uses paged allocation, by default
        False.
    """
    data: h5py.File | h5py.Group
    flush_interval: int = 100
    counter: FlushCounter = field(default_factory=FlushCounter)
    lock: threading.RLock = field(default_factory=threading.RLock)
    path: str | None = None
    access_mode: HDF5AccessMode | None = None
    file_options: HDF5FileOptions = field(default_factory=HDF5FileOptions)
    is_page_buffering_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate constructor arguments after dataclass initialization."""
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be greater than 0.")

    def store(
        self,
        keys: list[str],
        name: str,
        data: Any,
        is_dtype_change_enabled: bool = False,
    ) -> None:
        """Store a value in the HDF5 file at the specified location.

        Parameters
        ----------
        keys : list[str]
            List of group keys to navigate to the target subgroup.
        name : str
            Name of the dataset to store.
        data : Any
            Data to store in the dataset.
        is_dtype_change_enabled : bool, optional
            Whether to allow dtype changes when overwriting, by default False.

        Notes
        -----
        This method is thread-safe via internal locking.
        """
        with self.lock:
            subgroup = self.access_subgroup(keys=keys)
            subgroup.set_data(
                name=name,
                data=data,
                is_dtype_change_enabled=is_dtype_change_enabled,
            )
            subgroup.process_flush()

    def set_data(
        self,
        name: str,
        data: Any,
        is_dtype_change_enabled: bool = False,
    ) -> None:
        """Set data in the specified HDF5 group.

        Parameters
        ----------
        name : str
            Name of the dataset.
        data : Any
            Data to store.
        is_dtype_change_enabled : bool, optional
            Whether to allow dtype changes when overwriting, by default False.

        Notes
        -----
        This method is thread-safe via internal locking.
        """
        with self.lock:
            if type(data) != np.ndarray:
                data = np.array(data)

            if np.issubdtype(data.dtype, np.str_):
                data = data.astype('S')

            if name not in self.data.keys():
                self.data.create_dataset(
                    name=name,
                    dtype=data.dtype,
                    data=data,
                )
                return

            self._replace_data(
                data=data,
                name=name,
                is_dtype_change_enabled=is_dtype_change_enabled,
            )

    def _replace_data(
        self,
        data: np.ndarray,
        name: str,
        is_dtype_change_enabled: bool = False,
    ) -> None:
        """Replace an existing dataset value in the current group.

        Parameters
        ----------
        data : np.ndarray
            Data to store.
        name : str
            Name of the dataset.
        is_dtype_change_enabled : bool, optional
            Whether to allow dtype changes when overwriting, by default False.

        Raises
        ------
        ValueError
            If the target key does not resolve to a dataset.
        TypeError
            If dtype differs and dtype change is not enabled.

        Notes
        -----
        This method is thread-safe via internal locking. Can be called from
        locked or unlocked contexts due to RLock usage.
        """
        with self.lock:
            past_data = self.data[name]
            if not isinstance(past_data, h5py.Dataset):
                raise ValueError(f"Dataset {name} is not a dataset.")

            is_same_type = data.dtype == past_data.dtype
            is_same_shape = data.shape == past_data.shape

            if is_same_type and is_same_shape:
                past_data[()] = data
                return

            if not is_dtype_change_enabled and not is_same_type:
                logger.debug("dtype mismatch in group %s", self.data)
                raise TypeError(
                    f"Cannot overwrite data with different dtype.\n"
                    + f"Existing: {past_data.dtype}, New: {data.dtype}"
                )

            del self.data[name]
            self.data.create_dataset(
                name=name,
                dtype=data.dtype,
                data=data,
            )

    def access_value(
        self,
        keys: list[str],
        name: str,
    ) -> Any:
        """Retrieve a value from the HDF5 file at the specified location.

        Parameters
        ----------
        keys : list[str]
            List of group keys to navigate to the target subgroup.
        name : str
            Name of the dataset to retrieve.

        Returns
        -------
        Any
            The value stored in the dataset, or None if not found.

        Notes
        -----
        This method is thread-safe via internal locking.
        """
        with self.lock:
            subgroup = self.access_subgroup(keys=keys)
            return subgroup.get(name=name)

    def access_subgroup(
        self,
        keys: list[str],
    ) -> HDF5Container:
        """Access or create a subgroup using the provided key path.

        On a read-only container, navigating to an already-existing group
        is permitted and never creates anything; only a missing key raises,
        since creating it would require write access.

        Parameters
        ----------
        keys : list[str]
            List of group keys to navigate/create the subgroup.

        Returns
        -------
        HDF5Container
            The HDF5Container object containing the target subgroup.

        Raises
        ------
        PermissionError
            If this container's access mode is read-only and one of
            ``keys`` does not resolve to an existing group, so creating it
            would be required. Use :meth:`open_subgroup` to probe for a
            subgroup's existence without raising.

        Notes
        -----
        This method is thread-safe via internal locking. The returned
        container shares the parent's lock for coordinated multi-threaded access.
        """
        with self.lock:
            is_read_only = self.access_mode is not None and not self.access_mode.is_writable
            subgroup = self.data
            for key in keys:
                if is_read_only:
                    child = subgroup.get(key)
                    if not isinstance(child, h5py.Group):
                        raise PermissionError(
                            f"Cannot create group {key!r} on a read-only container; "
                            + "use open_subgroup to probe for it instead."
                        )
                    subgroup = child
                else:
                    subgroup = subgroup.require_group(key)
            return self._wrap(data=subgroup)

    def open_subgroup(self, keys: list[str]) -> HDF5Container | None:
        """Resolve an existing subgroup without creating anything.

        Parameters
        ----------
        keys : list[str]
            List of group keys to navigate to the target subgroup.

        Returns
        -------
        HDF5Container | None
            The container wrapping the resolved subgroup, or ``None``
            when any key in ``keys`` is missing or resolves to a dataset
            rather than a group.

        Notes
        -----
        This method is thread-safe via internal locking. Unlike
        :meth:`access_subgroup`, it never creates a group and is
        permitted on a read-only container.
        """
        with self.lock:
            node: h5py.File | h5py.Group = self.data
            for key in keys:
                child = node.get(key)
                if not isinstance(child, h5py.Group):
                    return None
                node = child
            return self._wrap(data=node)

    def _wrap(self, data: h5py.File | h5py.Group) -> HDF5Container:
        """Wrap a group or file node, propagating this container's shared state.

        Parameters
        ----------
        data : h5py.File | h5py.Group
            Node to wrap.

        Returns
        -------
        HDF5Container
            A new container sharing this container's counter, lock and
            file-level metadata.
        """
        return self.__class__(
            data=data,
            flush_interval=self.flush_interval,
            counter=self.counter,
            lock=self.lock,
            path=self.path,
            access_mode=self.access_mode,
            file_options=self.file_options,
            is_page_buffering_enabled=self.is_page_buffering_enabled,
        )

    def get(
        self,
        name: str,
    ) -> Any:
        """Retrieve a value from the specified HDF5 group.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        Any
            The dataset value if found, None otherwise.

        Notes
        -----
        This method is thread-safe via internal locking.
        """
        with self.lock:
            value = self.data.get(name, None)
            if isinstance(value, h5py.Dataset):
                output: Any = value[()]
                if isinstance(output, bytes):
                    output = output.decode('utf-8')
                return output
            return value

    def process_flush(self) -> None:
        """Flush periodically based on operation count.

        This increments the write counter and flushes when the configured
        flush_interval boundary is reached.

        Notes
        -----
        This method is typically called internally after each write operation.
        This method is thread-safe via internal locking.
        """
        with self.lock:
            self.counter.increment()
            if self.counter.is_flush_timing(flush_interval=self.flush_interval):
                self.flush()

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over key-value pairs in the current group.

        Group values are wrapped as HDF5Container objects and dataset values
        are returned as decoded Python objects.

        Notes
        -----
        This method is thread-safe via snapshot isolation. It takes a snapshot
        of the group structure within a lock, then yields items without holding
        the lock to prevent long-duration blocking of other threads.
        """
        with self.lock:
            items_snapshot = list(self.data.items())

        for key, value in items_snapshot:
            if isinstance(value, h5py.Group):
                yield key, self._wrap(data=value)
            elif isinstance(value, h5py.Dataset):
                yield key, self.get(name=key)
            else:
                yield key, value

    def values(self) -> Iterator[Any]:
        """Iterate over values in the current group.

        Group values are wrapped as HDF5Container objects and dataset values
        are returned as decoded Python objects.

        Returns
        -------
        Iterator[Any]
            Iterator of group wrappers or decoded dataset values.

        Notes
        -----
        This method is thread-safe via snapshot isolation. It takes a snapshot
        of the group structure within a lock, then yields items without holding
        the lock to prevent long-duration blocking of other threads.
        """
        with self.lock:
            values_snapshot = list(self.data.values())

        for value in values_snapshot:
            if isinstance(value, h5py.Group):
                yield self._wrap(data=value)
            elif isinstance(value, h5py.Dataset):
                output = value[()]
                if isinstance(output, bytes):
                    output = output.decode("utf-8")
                if isinstance(output, np.ndarray) and output.dtype.kind in {"S", "U"}:
                    output = output.astype(np.str_)
                yield output
            else:
                yield value

    def keys(self) -> Iterator[str]:
        """Iterate over keys in the current group.

        Returns
        -------
        Iterator[str]
            Key iterator for datasets/groups under the current node.

        Notes
        -----
        This method is thread-safe via snapshot isolation. It takes a snapshot
        of keys within a lock, then returns an iterator over the snapshot.
        """
        with self.lock:
            keys_snapshot: list[str] = list(self.data.keys())
        return iter(keys_snapshot)
