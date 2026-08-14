"""Specification of one column in a column store table."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColumnSpec:
    """Static shape and chunking specification for one growable column.

    Parameters
    ----------
    name : str
        Dataset name of the column within its group.
    row_shape : tuple[int, ...]
        Shape of a single row, ``()`` for a scalar column.
    dtype : np.dtype[np.generic]
        NumPy dtype of one element of the column.
    target_chunk_bytes : int, optional
        Approximate uncompressed byte size of one chunk, by default 1 MiB.
        The actual chunk row count is clamped to ``[1, 65536]``.

    Raises
    ------
    ValueError
        If ``name`` is empty, if any dimension of ``row_shape`` is not
        positive, or if the resulting row byte size is not positive.
    """

    name: str
    row_shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    target_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        """Validate the column specification."""
        if not self.name:
            raise ValueError("name must not be empty.")
        if any(dimension <= 0 for dimension in self.row_shape):
            raise ValueError(f"row_shape dimensions must be positive, got {self.row_shape}.")
        if self.row_bytes <= 0:
            raise ValueError(f"row_bytes must be positive, got {self.row_bytes}.")

    @property
    def row_bytes(self) -> int:
        """Return the byte size of one row.

        Returns
        -------
        int
            Number of bytes occupied by a single row.
        """
        element_count = math.prod(self.row_shape)
        return element_count * self.dtype.itemsize

    @property
    def rows_per_chunk(self) -> int:
        """Return the number of rows stored in one chunk.

        Returns
        -------
        int
            ``target_chunk_bytes // row_bytes`` clamped to ``[1, 65536]``.
            A small ``target_chunk_bytes`` on a wide column is honored down
            to one row per chunk, so a scattered single-row read only
            materializes that one row's chunk rather than a much larger
            one; callers writing sequentially should raise
            ``target_chunk_bytes`` instead of relying on a large minimum.
        """
        raw_rows_per_chunk = self.target_chunk_bytes // self.row_bytes
        return min(max(raw_rows_per_chunk, 1), 65536)

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the HDF5 chunk shape for this column's dataset.

        Returns
        -------
        tuple[int, ...]
            ``(rows_per_chunk, *row_shape)``.
        """
        return (self.rows_per_chunk, *self.row_shape)

    @property
    def max_shape(self) -> tuple[int | None, ...]:
        """Return the HDF5 maximum shape for this column's dataset.

        Returns
        -------
        tuple[int | None, ...]
            ``(None, *row_shape)``, unbounded along the row axis.
        """
        return (None, *self.row_shape)
