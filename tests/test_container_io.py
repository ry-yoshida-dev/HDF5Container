import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdf5_container import HDF5AccessMode, HDF5Container, HDF5FileOptions


class TestHDF5ContainerFromPath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.hdf5")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_from_path_creates_missing_parent_directory(self) -> None:
        nested_path = str(Path(self.tmpdir.name) / "nested" / "dir" / "test.hdf5")
        container = HDF5Container.from_path(path=nested_path)
        self.assertTrue(Path(nested_path).exists())
        container.close()

    def test_from_path_exposes_path_and_access_mode(self) -> None:
        container = HDF5Container.from_path(path=self.path)
        self.assertEqual(container.path, self.path)
        self.assertEqual(container.access_mode, HDF5AccessMode.APPEND)
        container.close()

    def test_from_path_read_only_on_missing_file_still_creates_it(self) -> None:
        container = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        self.assertTrue(Path(self.path).exists())
        self.assertEqual(container.access_mode, HDF5AccessMode.READ_ONLY)
        container.close()

    def test_from_path_reopening_existing_file_preserves_data(self) -> None:
        writer = HDF5Container.from_path(path=self.path)
        writer.store(keys=["group1"], name="x", data=123)
        writer.close()

        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        subgroup = reader.open_subgroup(keys=["group1"])
        self.assertIsNotNone(subgroup)
        assert subgroup is not None
        self.assertEqual(subgroup.get(name="x"), 123)
        reader.close()

    def test_from_path_with_non_paged_file_options(self) -> None:
        from hdf5_container import HDF5FileSpaceStrategy

        options = HDF5FileOptions(file_space_strategy=HDF5FileSpaceStrategy.FSM_AGGR)
        container = HDF5Container.from_path(path=self.path, file_options=options)
        self.assertFalse(container.is_page_buffering_enabled)
        container.store(keys=["group1"], name="x", data=1)
        container.close()


class TestOpenSubgroup(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.hdf5")
        self.container = HDF5Container.from_path(path=self.path, flush_interval=1)

    def tearDown(self) -> None:
        self.container.close()
        self.tmpdir.cleanup()

    def test_open_subgroup_returns_none_for_missing_group(self) -> None:
        result = self.container.open_subgroup(keys=["does_not_exist"])
        self.assertIsNone(result)
        self.assertNotIn("does_not_exist", list(self.container.keys()))

    def test_open_subgroup_returns_none_for_dataset(self) -> None:
        self.container.store(keys=[], name="scalar", data=1)
        result = self.container.open_subgroup(keys=["scalar"])
        self.assertIsNone(result)

    def test_open_subgroup_resolves_existing_group(self) -> None:
        self.container.store(keys=["group1"], name="x", data=42)
        subgroup = self.container.open_subgroup(keys=["group1"])
        self.assertIsNotNone(subgroup)
        assert subgroup is not None
        self.assertEqual(subgroup.get(name="x"), 42)


class TestReadOnlyEnforcement(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.hdf5")
        writer = HDF5Container.from_path(path=self.path)
        writer.store(keys=["group1"], name="x", data=1)
        writer.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_access_subgroup_raises_permission_error_for_missing_key_when_read_only(self) -> None:
        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        with self.assertRaises(PermissionError):
            reader.access_subgroup(keys=["does_not_exist"])
        reader.close()

    def test_access_subgroup_navigates_existing_group_when_read_only(self) -> None:
        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        subgroup = reader.access_subgroup(keys=["group1"])
        self.assertEqual(subgroup.get(name="x"), 1)
        reader.close()

    def test_open_subgroup_is_allowed_when_read_only(self) -> None:
        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        subgroup = reader.open_subgroup(keys=["group1"])
        self.assertIsNotNone(subgroup)
        reader.close()

    def test_store_still_fails_on_read_only_container(self) -> None:
        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        with self.assertRaises(ValueError):
            reader.store(keys=["group1"], name="y", data=2)
        reader.close()

    def test_set_data_still_fails_on_read_only_container(self) -> None:
        reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        subgroup = reader.access_subgroup(keys=["group1"])
        with self.assertRaises(ValueError):
            subgroup.set_data(name="y", data=2)
        reader.close()

    def test_multiple_concurrent_readers_share_one_prebuilt_cache_file(self) -> None:
        first_reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        second_reader = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        first_subgroup = first_reader.open_subgroup(keys=["group1"])
        second_subgroup = second_reader.open_subgroup(keys=["group1"])
        assert first_subgroup is not None
        assert second_subgroup is not None
        self.assertEqual(first_subgroup.get(name="x"), 1)
        self.assertEqual(second_subgroup.get(name="x"), 1)
        first_reader.close()
        second_reader.close()


class TestReopen(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "test.hdf5")
        writer = HDF5Container.from_path(path=self.path)
        writer.store(keys=["group1"], name="x", data=1)
        writer.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_reopen_promotes_read_only_to_append(self) -> None:
        container = HDF5Container.from_path(path=self.path, access_mode=HDF5AccessMode.READ_ONLY)
        with self.assertRaises(PermissionError):
            container.access_subgroup(keys=["group2"])

        container.reopen(access_mode=HDF5AccessMode.APPEND)
        self.assertEqual(container.access_mode, HDF5AccessMode.APPEND)
        container.store(keys=["group2"], name="y", data=2)
        self.assertEqual(container.access_value(keys=["group2"], name="y"), 2)
        container.close()

    def test_reopen_raises_type_error_for_group_backed_container(self) -> None:
        container = HDF5Container.from_path(path=self.path)
        subgroup = container.access_subgroup(keys=["group1"])
        with self.assertRaises(TypeError):
            subgroup.reopen(access_mode=HDF5AccessMode.READ_ONLY)
        container.close()

    def test_reopen_raises_type_error_without_path(self) -> None:
        import h5py

        raw_file = h5py.File(self.path, "a")
        container = HDF5Container(data=raw_file)
        with self.assertRaises(TypeError):
            container.reopen(access_mode=HDF5AccessMode.READ_ONLY)
        container.close()


if __name__ == "__main__":
    unittest.main()
