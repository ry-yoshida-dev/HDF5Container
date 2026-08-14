"""A group of columns sharing one logical row count."""

from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import NDArray

from .column_spec import ColumnSpec
from .growable_column import GrowableColumn

_ROW_COUNT_ATTRIBUTE = "row_count"
_DEFAULT_GROWTH_QUANTUM = 8192


class ColumnTable:
    """A set of :class:`GrowableColumn` objects sharing one logical length.

    The logical row count is the single source of truth for how many rows
    are actually present; it is stored as the ``row_count`` attribute on
    the backing group and is only ever advanced after every column's data
    has been written, so a crash mid-append leaves the table exactly as it
    was before the append started.

    Parameters
    ----------
    group : h5py.Group
        Group the columns live in.
    columns : dict[str, GrowableColumn]
        Columns keyed by name.
    growth_quantum : int
        Row-count granularity new capacity is reserved in.
    """

    def __init__(self, group: h5py.Group, columns: dict[str, GrowableColumn], growth_quantum: int) -> None:
        self._group = group
        self._columns = columns
        self._growth_quantum = growth_quantum

    @classmethod
    def create(
        cls,
        group: h5py.Group,
        specs: list[ColumnSpec],
        growth_quantum: int = _DEFAULT_GROWTH_QUANTUM,
    ) -> ColumnTable:
        """Create a new column table with a fresh dataset per column.

        Parameters
        ----------
        group : h5py.Group
            Group the columns are created in.
        specs : list[ColumnSpec]
            Specification of every column in the table.
        growth_quantum : int, optional
            Row-count granularity new capacity is reserved in, by default
            8192.

        Returns
        -------
        ColumnTable
            The newly created, empty column table.

        Raises
        ------
        ValueError
            If ``specs`` is empty or ``growth_quantum`` is not positive.
        """
        if not specs:
            raise ValueError("specs must not be empty.")
        if growth_quantum <= 0:
            raise ValueError(f"growth_quantum must be positive, got {growth_quantum}.")
        columns = {
            spec.name: GrowableColumn.create(group=group, spec=spec, initial_capacity=growth_quantum)
            for spec in specs
        }
        group.attrs[_ROW_COUNT_ATTRIBUTE] = 0
        return cls(group=group, columns=columns, growth_quantum=growth_quantum)

    @classmethod
    def open(cls, group: h5py.Group, specs: list[ColumnSpec]) -> ColumnTable:
        """Open an existing column table.

        Parameters
        ----------
        group : h5py.Group
            Group the columns live in.
        specs : list[ColumnSpec]
            Specification of every column expected in the table, each
            validated against the corresponding existing dataset.

        Returns
        -------
        ColumnTable
            The opened column table.

        Raises
        ------
        ValueError
            If ``specs`` is empty.

        Notes
        -----
        Newly reserved capacity after opening uses the default growth
        quantum, since the growth quantum used at creation time is not
        persisted.
        """
        if not specs:
            raise ValueError("specs must not be empty.")
        columns = {spec.name: GrowableColumn.open(group=group, spec=spec) for spec in specs}
        return cls(group=group, columns=columns, growth_quantum=_DEFAULT_GROWTH_QUANTUM)

    @property
    def row_count(self) -> int:
        """Return the current logical row count.

        Returns
        -------
        int
            Number of rows currently visible in the table.

        Raises
        ------
        TypeError
            If the stored ``row_count`` attribute is not an integer.
        """
        row_count = self._group.attrs[_ROW_COUNT_ATTRIBUTE]
        if not isinstance(row_count, int | np.integer):
            raise TypeError(f"row_count attribute must be an integer, got {type(row_count)!r}.")
        return int(row_count)

    @property
    def column_names(self) -> list[str]:
        """Return the names of every column in the table.

        Returns
        -------
        list[str]
            Column names.
        """
        return list(self._columns.keys())

    def append(self, rows: dict[str, NDArray[np.generic]]) -> int:
        """Append rows to every column and advance the logical row count.

        Parameters
        ----------
        rows : dict[str, NDArray[np.generic]]
            Row arrays keyed by column name. Must supply exactly the
            table's columns, all with equal leading length.

        Returns
        -------
        int
            The row index the appended rows start at.

        Raises
        ------
        ValueError
            If ``rows`` does not supply exactly the table's columns, or if
            the supplied arrays do not share the same leading length.
        """
        if set(rows.keys()) != set(self._columns.keys()):
            raise ValueError(
                f"rows must supply exactly the table's columns. expected={sorted(self._columns.keys())}, "
                + f"got={sorted(rows.keys())}."
            )
        row_lengths = {name: array.shape[0] for name, array in rows.items()}
        if len(set(row_lengths.values())) != 1:
            raise ValueError(f"all columns must have equal leading length, got {row_lengths}.")
        start = self.row_count
        appended_row_count = next(iter(row_lengths.values()))
        required_capacity = start + appended_row_count
        for column in self._columns.values():
            column.reserve(required_capacity=required_capacity, growth_quantum=self._growth_quantum)
        for name, column in self._columns.items():
            column.write(start=start, rows=rows[name])
        self._group.attrs[_ROW_COUNT_ATTRIBUTE] = required_capacity
        return start

    def read_slice(self, start: int, count: int) -> dict[str, NDArray[np.generic]]:
        """Read a contiguous slice of every column.

        Parameters
        ----------
        start : int
            Row index to start reading at.
        count : int
            Number of rows to read.

        Returns
        -------
        dict[str, NDArray[np.generic]]
            Column arrays keyed by column name.
        """
        return {name: column.read(start=start, count=count) for name, column in self._columns.items()}

    def read_column_slice(self, name: str, start: int, count: int) -> NDArray[np.generic]:
        """Read a contiguous slice of one column.

        Parameters
        ----------
        name : str
            Column name.
        start : int
            Row index to start reading at.
        count : int
            Number of rows to read.

        Returns
        -------
        NDArray[np.generic]
            The requested rows of the named column.
        """
        return self._resolve_column(name).read(start=start, count=count)

    def read_indices(self, indices: NDArray[np.int64]) -> dict[str, NDArray[np.generic]]:
        """Read arbitrary row indices from every column.

        Parameters
        ----------
        indices : NDArray[np.int64]
            Row indices to read, in any order and possibly with
            duplicates.

        Returns
        -------
        dict[str, NDArray[np.generic]]
            Column arrays keyed by column name, in the order of
            ``indices``.
        """
        return {name: column.read_indices(indices=indices) for name, column in self._columns.items()}

    def read_column_indices(self, name: str, indices: NDArray[np.int64]) -> NDArray[np.generic]:
        """Read arbitrary row indices from one column.

        Parameters
        ----------
        name : str
            Column name.
        indices : NDArray[np.int64]
            Row indices to read, in any order and possibly with
            duplicates.

        Returns
        -------
        NDArray[np.generic]
            Rows of the named column, in the order of ``indices``.
        """
        return self._resolve_column(name).read_indices(indices=indices)

    def write_column_indices(self, name: str, indices: NDArray[np.int64], values: NDArray[np.generic]) -> None:
        """Write arbitrary row indices of one column.

        Parameters
        ----------
        name : str
            Column name.
        indices : NDArray[np.int64]
            Row indices to write, in any order and possibly with
            duplicates.
        values : NDArray[np.generic]
            Values to write, aligned with ``indices``.
        """
        self._resolve_column(name).write_indices(indices=indices, values=values)

    def truncate_to_row_count(self) -> None:
        """Shrink every column's capacity down to the logical row count.

        Call this when closing a writer to release capacity that was
        reserved ahead of writes but never logically committed.
        """
        row_count = self.row_count
        for column in self._columns.values():
            column.truncate(row_count=row_count)

    def _resolve_column(self, name: str) -> GrowableColumn:
        """Look up a column by name.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        GrowableColumn
            The named column.

        Raises
        ------
        KeyError
            If no column named ``name`` exists in the table.
        """
        if name not in self._columns:
            raise KeyError(f"Unknown column: {name!r}.")
        return self._columns[name]
