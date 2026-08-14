from __future__ import annotations

import os
from typing import TypeVar

import h5py

from ..options import HDF5AccessMode, HDF5FileOptions
from .protocols import HDF5ContainerProtocol

TContainer = TypeVar("TContainer", bound=HDF5ContainerProtocol)

_PAGE_STRATEGY_CODE = h5py.h5f.FSPACE_STRATEGY_PAGE


class IOMixin:
    """I/O behavior for file-backed HDF5 containers."""

    @classmethod
    def from_path(
        cls: type[TContainer],
        path: str,
        flush_interval: int = 100,
        access_mode: HDF5AccessMode = HDF5AccessMode.APPEND,
        file_options: HDF5FileOptions | None = None,
    ) -> TContainer:
        """Create a container from an HDF5 file path.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to open/create.
        flush_interval : int, optional
            Number of write operations between automatic flushes.
        access_mode : HDF5AccessMode, optional
            Mode the file is opened in, by default
            :attr:`HDF5AccessMode.APPEND`.
        file_options : HDF5FileOptions | None, optional
            Tuning options applied at creation and open time. Defaults to
            :class:`HDF5FileOptions` with its default values.

        Returns
        -------
        TContainer
            A container instance backed by the target HDF5 file.

        Notes
        -----
        When the file does not yet exist, it is created with
        ``file_options.creation_kwargs()``. A concurrent creator winning
        the race to create the same path is tolerated: the
        :class:`FileExistsError` is only re-raised when the file is still
        missing afterwards.
        """
        resolved_file_options = file_options if file_options is not None else HDF5FileOptions()
        parent_directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_directory, exist_ok=True)
        if not os.path.exists(path):
            IOMixin._create_file(path=path, file_options=resolved_file_options)
        is_page_buffering_enabled = IOMixin._probe_is_page_buffering_enabled(
            path=path, is_locking_disabled=not access_mode.is_writable
        )
        open_kwargs = resolved_file_options.open_kwargs(is_page_buffering_enabled=is_page_buffering_enabled)
        if not access_mode.is_writable:
            open_kwargs["locking"] = False
        data = IOMixin._open_file(path=path, mode=access_mode.h5py_mode, kwargs=open_kwargs)
        return cls(  # type: ignore[call-arg]
            data=data,
            flush_interval=flush_interval,
            path=path,
            access_mode=access_mode,
            file_options=resolved_file_options,
            is_page_buffering_enabled=is_page_buffering_enabled,
        )

    @staticmethod
    def _create_file(path: str, file_options: HDF5FileOptions) -> None:
        """Create a new HDF5 file, tolerating a concurrent creator.

        Parameters
        ----------
        path : str
            Path of the file to create.
        file_options : HDF5FileOptions
            Options controlling the creation-time file layout.

        Raises
        ------
        FileExistsError
            If the file still does not exist after this creation attempt
            failed, meaning the failure was not caused by a concurrent
            creator.
        """
        try:
            creation_file = IOMixin._open_file(path=path, mode="x", kwargs=file_options.creation_kwargs())
            creation_file.close()
        except FileExistsError:
            if not os.path.exists(path):
                raise

    @staticmethod
    def _open_file(path: str, mode: str, kwargs: dict[str, object]) -> h5py.File:
        """Open or create an HDF5 file from a dynamically built keyword mapping.

        Parameters
        ----------
        path : str
            Path of the file to open or create.
        mode : str
            h5py file mode, for example ``"x"``, ``"r"`` or ``"a"``.
        kwargs : dict[str, object]
            Tuning keyword arguments produced by :class:`HDF5FileOptions`.
            Their values are heterogeneous by construction, which is
            statically wider than ``h5py.File``'s per-keyword parameter
            types, so this is the single, deliberately isolated point
            where that widening is accepted.

        Returns
        -------
        h5py.File
            The opened file handle.
        """
        return h5py.File(path, mode, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _probe_is_page_buffering_enabled(path: str, is_locking_disabled: bool) -> bool:
        """Detect whether an existing file was created with paged allocation.

        Parameters
        ----------
        path : str
            Path of the file to probe.
        is_locking_disabled : bool
            Whether to open the probe with ``locking=False``. This must
            match the locking behavior of the read/write open that will
            follow this probe in the same call, since HDF5 rejects
            opening a file with a locking flag that differs from another
            handle already open on it. Pass ``True`` when the container
            is being opened read-only.

        Returns
        -------
        bool
            ``True`` when the file's file-space strategy is
            :attr:`~hdf5_container.options.HDF5FileSpaceStrategy.PAGE`.

        Notes
        -----
        Read-only opens use ``locking=False`` so that several processes
        can share one prebuilt, read-only cache file without file locking
        making this probe fail or block when another reader holds the
        file open.
        """
        with h5py.File(path, "r", locking=False if is_locking_disabled else None) as probe_file:
            strategy_code = probe_file.id.get_create_plist().get_file_space_strategy()[0]
            return strategy_code == _PAGE_STRATEGY_CODE

    def reopen(self: HDF5ContainerProtocol, access_mode: HDF5AccessMode) -> None:
        """Close and reopen this container's file under a new access mode.

        Parameters
        ----------
        access_mode : HDF5AccessMode
            The access mode to reopen the file with.

        Raises
        ------
        TypeError
            If this container does not directly wrap an ``h5py.File``
            (for example, a container returned by ``access_subgroup`` or
            ``open_subgroup``), or if it was not created via
            :meth:`from_path` and therefore has no known ``path``.

        Notes
        -----
        This method is thread-safe via internal locking. Any container
        previously obtained from this container's ``access_subgroup`` or
        ``open_subgroup`` becomes stale once the file is reopened, since
        its underlying ``h5py.Group`` handle belongs to the closed file.
        Callers must re-resolve such subgroup containers afterwards.
        """
        with self.lock:
            if not isinstance(self.data, h5py.File):
                raise TypeError("reopen requires a container that directly wraps an h5py.File.")
            if self.path is None:
                raise TypeError("reopen requires a container created via from_path.")
            self.data.flush()
            self.data.close()
            open_kwargs = self.file_options.open_kwargs(
                is_page_buffering_enabled=self.is_page_buffering_enabled
            )
            if not access_mode.is_writable:
                open_kwargs["locking"] = False
            self.data = IOMixin._open_file(path=self.path, mode=access_mode.h5py_mode, kwargs=open_kwargs)
            self.access_mode = access_mode

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
