"""HDF5 file creation and open tuning options."""

from __future__ import annotations

from dataclasses import dataclass

from .file_space_strategy import HDF5FileSpaceStrategy
from .library_version import HDF5LibraryVersion


def _is_power_of_two(value: int) -> bool:
    """Return whether a positive integer is a power of two.

    Parameters
    ----------
    value : int
        The integer to test.

    Returns
    -------
    bool
        ``True`` when ``value`` is a power of two.
    """
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class HDF5FileOptions:
    """Tuning knobs for creating and (re)opening an HDF5 file.

    These options are split into two groups: the ones that only take
    effect at file creation time (library version bound, file space
    strategy and page size) and the ones that tune runtime caching and
    apply on every open (chunk cache, page buffer, alignment).

    Parameters
    ----------
    library_version : HDF5LibraryVersion, optional
        Lower bound on the on-disk object format, by default
        :attr:`HDF5LibraryVersion.LATEST`.
    file_space_strategy : HDF5FileSpaceStrategy, optional
        File space management strategy used at creation time, by default
        :attr:`HDF5FileSpaceStrategy.PAGE`.
    file_space_page_size : int, optional
        Size in bytes of one file space page, by default 4 MiB. Must be a
        power of two of at least 512 bytes.
    page_buffer_size : int, optional
        Size in bytes of the page buffer, by default 64 MiB. Must be a
        multiple of ``file_space_page_size``.
    chunk_cache_bytes : int, optional
        Raw chunk cache size in bytes (``rdcc_nbytes``), by default 256 MiB.
    chunk_cache_slots : int, optional
        Number of chunk cache slots (``rdcc_nslots``), by default 20011.
    chunk_cache_preemption : float, optional
        Chunk cache preemption policy (``rdcc_w0``) in ``[0, 1]``, by
        default 0.75.
    alignment_threshold : int, optional
        Minimum allocation size in bytes subject to alignment, by default
        4096.
    alignment_interval : int, optional
        Alignment boundary in bytes for allocations at or above
        ``alignment_threshold``, by default 4096.
    """

    library_version: HDF5LibraryVersion = HDF5LibraryVersion.LATEST
    file_space_strategy: HDF5FileSpaceStrategy = HDF5FileSpaceStrategy.PAGE
    file_space_page_size: int = 4 * 1024 * 1024
    page_buffer_size: int = 64 * 1024 * 1024
    chunk_cache_bytes: int = 256 * 1024 * 1024
    chunk_cache_slots: int = 20011
    chunk_cache_preemption: float = 0.75
    alignment_threshold: int = 4096
    alignment_interval: int = 4096

    def __post_init__(self) -> None:
        """Validate the tuning options.

        Raises
        ------
        ValueError
            If ``file_space_page_size`` is not a power of two of at least
            512 bytes, if ``page_buffer_size`` is not a multiple of
            ``file_space_page_size``, or if ``chunk_cache_preemption`` is
            outside ``[0, 1]``.
        """
        if self.file_space_page_size < 512 or not _is_power_of_two(self.file_space_page_size):
            raise ValueError(
                f"file_space_page_size must be a power of two >= 512, got {self.file_space_page_size}."
            )
        if self.page_buffer_size % self.file_space_page_size != 0:
            raise ValueError(
                "page_buffer_size must be a multiple of file_space_page_size, "
                + f"got page_buffer_size={self.page_buffer_size}, "
                + f"file_space_page_size={self.file_space_page_size}."
            )
        if not 0.0 <= self.chunk_cache_preemption <= 1.0:
            raise ValueError(
                f"chunk_cache_preemption must be in [0, 1], got {self.chunk_cache_preemption}."
            )

    def creation_kwargs(self) -> dict[str, object]:
        """Build the keyword arguments for creating a new HDF5 file.

        Returns
        -------
        dict[str, object]
            Keyword arguments suitable for ``h5py.File(path, "x", **kwargs)``.
        """
        return {
            "libver": self.library_version.bound,
            "fs_strategy": self.file_space_strategy.value,
            "fs_page_size": self.file_space_page_size,
            "fs_persist": True,
            "fs_threshold": 1,
            "rdcc_nbytes": self.chunk_cache_bytes,
            "rdcc_nslots": self.chunk_cache_slots,
            "rdcc_w0": self.chunk_cache_preemption,
            "alignment_threshold": self.alignment_threshold,
            "alignment_interval": self.alignment_interval,
        }

    def open_kwargs(self, is_page_buffering_enabled: bool) -> dict[str, object]:
        """Build the keyword arguments for opening an existing HDF5 file.

        Parameters
        ----------
        is_page_buffering_enabled : bool
            Whether the target file was created with paged allocation.
            ``page_buf_size`` must never be passed for a file that was not
            created with paged allocation.

        Returns
        -------
        dict[str, object]
            Keyword arguments suitable for ``h5py.File(path, mode, **kwargs)``.
        """
        kwargs: dict[str, object] = {
            "libver": self.library_version.bound,
            "rdcc_nbytes": self.chunk_cache_bytes,
            "rdcc_nslots": self.chunk_cache_slots,
            "rdcc_w0": self.chunk_cache_preemption,
            "alignment_threshold": self.alignment_threshold,
            "alignment_interval": self.alignment_interval,
        }
        if is_page_buffering_enabled:
            kwargs["page_buf_size"] = self.page_buffer_size
        return kwargs
