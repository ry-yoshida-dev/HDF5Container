"""One resizable HDF5 dataset backing a column store column."""

from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import NDArray

from .column_spec import ColumnSpec


class GrowableColumn:
    """A single resizable HDF5 dataset that owns storage capacity.

    A ``GrowableColumn`` tracks only the dataset's allocated capacity. The
    logical number of rows actually written is tracked by the owning
    :class:`~hdf5_container.table.column_table.ColumnTable`, not by this
    class, so that capacity can be grown ahead of writes without exposing
    unwritten rows as logically present.

    Parameters
    ----------
    dataset : h5py.Dataset
        The underlying resizable dataset.
    spec : ColumnSpec
        The static shape and chunking specification this dataset was
        created from or validated against.
    """

    def __init__(self, dataset: h5py.Dataset, spec: ColumnSpec) -> None:
        self._dataset = dataset
        self._spec = spec

    @classmethod
    def create(cls, group: h5py.Group, spec: ColumnSpec, initial_capacity: int) -> GrowableColumn:
        """Create a new resizable dataset for a column.

        Parameters
        ----------
        group : h5py.Group
            Group the dataset is created in.
        spec : ColumnSpec
            Shape, dtype and chunking specification for the column.
        initial_capacity : int
            Initial number of rows to allocate.

        Returns
        -------
        GrowableColumn
            A wrapper around the newly created dataset.

        Raises
        ------
        ValueError
            If ``initial_capacity`` is not positive.
        """
        if initial_capacity <= 0:
            raise ValueError(f"initial_capacity must be positive, got {initial_capacity}.")
        dataset = group.create_dataset(
            name=spec.name,
            shape=(initial_capacity, *spec.row_shape),
            maxshape=spec.max_shape,
            chunks=spec.chunk_shape,
            dtype=spec.dtype,
        )
        return cls(dataset=dataset, spec=spec)

    @classmethod
    def open(cls, group: h5py.Group, spec: ColumnSpec) -> GrowableColumn:
        """Open an existing dataset backing a column.

        Parameters
        ----------
        group : h5py.Group
            Group the dataset is expected to live in.
        spec : ColumnSpec
            Shape and dtype the existing dataset is validated against.

        Returns
        -------
        GrowableColumn
            A wrapper around the existing dataset.

        Raises
        ------
        ValueError
            If ``spec.name`` does not resolve to a dataset, or if the
            dataset's dtype or row shape does not match ``spec``.
        """
        node = group[spec.name]
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"{spec.name!r} does not resolve to a dataset.")
        if node.dtype != spec.dtype:
            raise ValueError(f"dtype mismatch for column {spec.name!r}: expected {spec.dtype}, got {node.dtype}.")
        if node.shape[1:] != spec.row_shape:
            raise ValueError(
                f"row_shape mismatch for column {spec.name!r}: expected {spec.row_shape}, got {node.shape[1:]}."
            )
        return cls(dataset=node, spec=spec)

    @property
    def spec(self) -> ColumnSpec:
        """Return the column specification this dataset conforms to.

        Returns
        -------
        ColumnSpec
            The column's static shape and chunking specification.
        """
        return self._spec

    @property
    def capacity(self) -> int:
        """Return the number of rows currently allocated on disk.

        Returns
        -------
        int
            The dataset's current size along the row axis.
        """
        return self._dataset.shape[0]

    def reserve(self, required_capacity: int, growth_quantum: int) -> None:
        """Grow the dataset so it can hold at least ``required_capacity`` rows.

        Parameters
        ----------
        required_capacity : int
            Minimum number of rows the dataset must be able to hold.
        growth_quantum : int
            Granularity to grow by. The dataset is resized to the smallest
            multiple of ``growth_quantum`` that is at least
            ``required_capacity``. Nothing happens when the current
            capacity already suffices.

        Raises
        ------
        ValueError
            If ``growth_quantum`` is not positive.
        """
        if growth_quantum <= 0:
            raise ValueError(f"growth_quantum must be positive, got {growth_quantum}.")
        if required_capacity <= self.capacity:
            return
        quanta_count = -(-required_capacity // growth_quantum)
        new_capacity = quanta_count * growth_quantum
        self._dataset.resize(new_capacity, axis=0)

    def write(self, start: int, rows: NDArray[np.generic]) -> None:
        """Write a contiguous block of rows starting at ``start``.

        Parameters
        ----------
        start : int
            Row index to start writing at.
        rows : NDArray[np.generic]
            Rows to write, with leading dimension equal to the row count
            and trailing dimensions equal to ``spec.row_shape``.
        """
        row_count = rows.shape[0]
        self._dataset[start : start + row_count] = rows

    def read(self, start: int, count: int) -> NDArray[np.generic]:
        """Read a contiguous block of ``count`` rows starting at ``start``.

        Parameters
        ----------
        start : int
            Row index to start reading at.
        count : int
            Number of rows to read.

        Returns
        -------
        NDArray[np.generic]
            The requested rows.
        """
        return self._dataset[start : start + count]

    def read_indices(self, indices: NDArray[np.int64]) -> NDArray[np.generic]:
        """Read arbitrary, possibly unordered and duplicated, row indices.

        h5py only accepts strictly increasing, duplicate-free fancy
        selections, so the requested indices are deduplicated and sorted
        before issuing a single dataset read, and the result is expanded
        back to match the caller's original order (duplicates included).

        Parameters
        ----------
        indices : NDArray[np.int64]
            Row indices to read, in any order and possibly with
            duplicates.

        Returns
        -------
        NDArray[np.generic]
            Rows in the same order as ``indices``.
        """
        sort_order = np.argsort(indices, kind="stable")
        sorted_indices = indices[sort_order]
        unique_indices, inverse = np.unique(sorted_indices, return_inverse=True)
        selected = self._dataset[unique_indices]
        gathered = selected[inverse]
        output = np.empty_like(gathered)
        output[sort_order] = gathered
        return output

    def write_indices(self, indices: NDArray[np.int64], values: NDArray[np.generic]) -> None:
        """Write values at arbitrary, possibly unordered and duplicated, row indices.

        When the same row index appears more than once, the value
        associated with its last occurrence in ``indices`` wins, matching
        ordinary NumPy fancy-index assignment semantics.

        Parameters
        ----------
        indices : NDArray[np.int64]
            Row indices to write, in any order and possibly with
            duplicates.
        values : NDArray[np.generic]
            Values to write, aligned with ``indices``.

        Raises
        ------
        ValueError
            If ``indices`` and ``values`` do not have the same leading
            length.
        """
        if indices.shape[0] != values.shape[0]:
            raise ValueError(
                f"indices and values must have equal leading length, got {indices.shape[0]} and {values.shape[0]}."
            )
        sort_order = np.argsort(indices, kind="stable")
        sorted_indices = indices[sort_order]
        sorted_values = values[sort_order]
        unique_indices, inverse = np.unique(sorted_indices, return_inverse=True)
        last_position_per_unique = np.zeros(unique_indices.shape[0], dtype=np.int64)
        last_position_per_unique[inverse] = np.arange(sorted_indices.shape[0])
        self._dataset[unique_indices] = sorted_values[last_position_per_unique]

    def truncate(self, row_count: int) -> None:
        """Shrink the dataset down to exactly ``row_count`` rows.

        Parameters
        ----------
        row_count : int
            Number of rows to keep.
        """
        self._dataset.resize(row_count, axis=0)
