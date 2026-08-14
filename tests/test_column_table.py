import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdf5_container import ColumnSpec, ColumnTable, GrowableColumn


class TestColumnSpec(unittest.TestCase):
    def test_row_bytes_for_vector_column(self) -> None:
        spec = ColumnSpec(name="box_coord", row_shape=(4,), dtype=np.dtype(np.float32))
        self.assertEqual(spec.row_bytes, 16)

    def test_row_bytes_for_scalar_column(self) -> None:
        spec = ColumnSpec(name="box_conf", row_shape=(), dtype=np.dtype(np.float32))
        self.assertEqual(spec.row_bytes, 4)

    def test_rows_per_chunk_is_clamped(self) -> None:
        huge_row_spec = ColumnSpec(
            name="feature", row_shape=(1_000_000,), dtype=np.dtype(np.float32), target_chunk_bytes=1024
        )
        self.assertEqual(huge_row_spec.rows_per_chunk, 1)

        tiny_row_spec = ColumnSpec(name="flag", row_shape=(), dtype=np.dtype(np.uint8), target_chunk_bytes=1024**3)
        self.assertEqual(tiny_row_spec.rows_per_chunk, 65536)

    def test_wide_column_with_small_target_chunk_bytes_gets_few_rows_per_chunk(self) -> None:
        feature_spec = ColumnSpec(
            name="feature", row_shape=(512,), dtype=np.dtype(np.float32), target_chunk_bytes=4096
        )
        self.assertEqual(feature_spec.row_bytes, 2048)
        self.assertEqual(feature_spec.rows_per_chunk, 2)
        self.assertEqual(feature_spec.chunk_shape, (2, 512))

        single_row_spec = ColumnSpec(
            name="feature", row_shape=(512,), dtype=np.dtype(np.float32), target_chunk_bytes=1
        )
        self.assertEqual(single_row_spec.rows_per_chunk, 1)

    def test_chunk_shape_and_max_shape(self) -> None:
        spec = ColumnSpec(name="box_coord", row_shape=(4,), dtype=np.dtype(np.float32))
        self.assertEqual(spec.chunk_shape, (spec.rows_per_chunk, 4))
        self.assertEqual(spec.max_shape, (None, 4))

    def test_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            ColumnSpec(name="", row_shape=(), dtype=np.dtype(np.float32))

    def test_rejects_non_positive_dimension(self) -> None:
        with self.assertRaises(ValueError):
            ColumnSpec(name="x", row_shape=(0,), dtype=np.dtype(np.float32))


class TestGrowableColumn(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.h5")
        self.file = h5py.File(self.path, "w")

    def tearDown(self) -> None:
        self.file.close()
        self.tmpdir.cleanup()

    def test_reserve_grows_to_quantum_multiple(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(3,), dtype=np.dtype(np.float32))
        column = GrowableColumn.create(group=self.file, spec=spec, initial_capacity=8)
        column.reserve(required_capacity=10, growth_quantum=8)
        self.assertEqual(column.capacity, 16)

    def test_reserve_does_nothing_when_capacity_suffices(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(3,), dtype=np.dtype(np.float32))
        column = GrowableColumn.create(group=self.file, spec=spec, initial_capacity=16)
        column.reserve(required_capacity=10, growth_quantum=8)
        self.assertEqual(column.capacity, 16)

    def test_read_indices_handles_unordered_and_duplicated(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(2,), dtype=np.dtype(np.float32))
        column = GrowableColumn.create(group=self.file, spec=spec, initial_capacity=10)
        data = np.arange(20, dtype=np.float32).reshape(10, 2)
        column.write(start=0, rows=data)

        indices = np.array([5, 0, 2, 2, 5, 9], dtype=np.int64)
        result = column.read_indices(indices=indices)
        expected = data[indices]
        np.testing.assert_array_equal(result, expected)

    def test_write_indices_last_occurrence_wins_for_duplicates(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(), dtype=np.dtype(np.int32))
        column = GrowableColumn.create(group=self.file, spec=spec, initial_capacity=5)
        column.write(start=0, rows=np.zeros(5, dtype=np.int32))

        indices = np.array([3, 1, 3], dtype=np.int64)
        values = np.array([100, 200, 300], dtype=np.int32)
        column.write_indices(indices=indices, values=values)

        result = column.read(start=0, count=5)
        np.testing.assert_array_equal(result, np.array([0, 200, 0, 300, 0], dtype=np.int32))

    def test_open_validates_dtype_mismatch(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(3,), dtype=np.dtype(np.float32))
        GrowableColumn.create(group=self.file, spec=spec, initial_capacity=8)

        mismatched_spec = ColumnSpec(name="x", row_shape=(3,), dtype=np.dtype(np.float64))
        with self.assertRaises(ValueError):
            GrowableColumn.open(group=self.file, spec=mismatched_spec)

    def test_open_validates_row_shape_mismatch(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(3,), dtype=np.dtype(np.float32))
        GrowableColumn.create(group=self.file, spec=spec, initial_capacity=8)

        mismatched_spec = ColumnSpec(name="x", row_shape=(4,), dtype=np.dtype(np.float32))
        with self.assertRaises(ValueError):
            GrowableColumn.open(group=self.file, spec=mismatched_spec)

    def test_truncate_shrinks_capacity(self) -> None:
        spec = ColumnSpec(name="x", row_shape=(), dtype=np.dtype(np.int32))
        column = GrowableColumn.create(group=self.file, spec=spec, initial_capacity=16)
        column.truncate(row_count=5)
        self.assertEqual(column.capacity, 5)


class TestColumnTable(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.h5")
        self.file = h5py.File(self.path, "w")
        self.specs = [
            ColumnSpec(name="box_coord", row_shape=(4,), dtype=np.dtype(np.float32)),
            ColumnSpec(name="box_conf", row_shape=(), dtype=np.dtype(np.float32)),
        ]

    def tearDown(self) -> None:
        self.file.close()
        self.tmpdir.cleanup()

    def test_append_and_read_slice_round_trip(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        box_coord = np.arange(12, dtype=np.float32).reshape(3, 4)
        box_conf = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        start = table.append(rows={"box_coord": box_coord, "box_conf": box_conf})
        self.assertEqual(start, 0)
        self.assertEqual(table.row_count, 3)

        result = table.read_slice(start=0, count=3)
        np.testing.assert_array_equal(result["box_coord"], box_coord)
        np.testing.assert_array_almost_equal(result["box_conf"], box_conf)

    def test_append_grows_across_quantum_boundary(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        for _ in range(3):
            table.append(
                rows={
                    "box_coord": np.zeros((3, 4), dtype=np.float32),
                    "box_conf": np.zeros(3, dtype=np.float32),
                }
            )
        self.assertEqual(table.row_count, 9)
        column_capacity = self.file["box_coord"].shape[0]
        self.assertGreaterEqual(column_capacity, 9)
        self.assertEqual(column_capacity % 4, 0)

    def test_append_rejects_mismatched_columns(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        with self.assertRaises(ValueError):
            table.append(rows={"box_coord": np.zeros((3, 4), dtype=np.float32)})

    def test_append_rejects_unequal_leading_lengths(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        with self.assertRaises(ValueError):
            table.append(
                rows={
                    "box_coord": np.zeros((3, 4), dtype=np.float32),
                    "box_conf": np.zeros(2, dtype=np.float32),
                }
            )

    def test_read_indices_unordered_and_duplicated_across_columns(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        box_coord = np.arange(40, dtype=np.float32).reshape(10, 4)
        box_conf = np.arange(10, dtype=np.float32)
        table.append(rows={"box_coord": box_coord, "box_conf": box_conf})

        indices = np.array([7, 1, 1, 9, 0], dtype=np.int64)
        result = table.read_indices(indices=indices)
        np.testing.assert_array_equal(result["box_coord"], box_coord[indices])
        np.testing.assert_array_equal(result["box_conf"], box_conf[indices])

    def test_write_column_indices(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        box_coord = np.zeros((5, 4), dtype=np.float32)
        box_conf = np.zeros(5, dtype=np.float32)
        table.append(rows={"box_coord": box_coord, "box_conf": box_conf})

        table.write_column_indices(
            name="box_conf",
            indices=np.array([4, 2], dtype=np.int64),
            values=np.array([9.5, 3.5], dtype=np.float32),
        )
        result = table.read_column_slice(name="box_conf", start=0, count=5)
        np.testing.assert_array_almost_equal(result, np.array([0, 0, 3.5, 0, 9.5], dtype=np.float32))

    def test_truncate_to_row_count_shrinks_capacity_to_logical_length(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        table.append(
            rows={
                "box_coord": np.zeros((3, 4), dtype=np.float32),
                "box_conf": np.zeros(3, dtype=np.float32),
            }
        )
        self.assertEqual(self.file["box_coord"].shape[0], 4)
        table.truncate_to_row_count()
        self.assertEqual(self.file["box_coord"].shape[0], 3)

    def test_open_reopens_existing_table(self) -> None:
        table = ColumnTable.create(group=self.file, specs=self.specs, growth_quantum=4)
        table.append(
            rows={
                "box_coord": np.ones((2, 4), dtype=np.float32),
                "box_conf": np.ones(2, dtype=np.float32),
            }
        )

        reopened = ColumnTable.open(group=self.file, specs=self.specs)
        self.assertEqual(reopened.row_count, 2)
        self.assertEqual(sorted(reopened.column_names), sorted(spec.name for spec in self.specs))


if __name__ == "__main__":
    unittest.main()
