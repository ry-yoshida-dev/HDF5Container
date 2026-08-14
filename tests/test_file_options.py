import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdf5_container import HDF5AccessMode, HDF5FileOptions, HDF5FileSpaceStrategy, HDF5LibraryVersion


class TestHDF5FileOptions(unittest.TestCase):
    def test_default_creation_kwargs(self) -> None:
        options = HDF5FileOptions()
        kwargs = options.creation_kwargs()
        self.assertEqual(kwargs["libver"], "latest")
        self.assertEqual(kwargs["fs_strategy"], "page")
        self.assertEqual(kwargs["fs_page_size"], 4 * 1024 * 1024)
        self.assertEqual(kwargs["fs_persist"], True)
        self.assertEqual(kwargs["fs_threshold"], 1)
        self.assertEqual(kwargs["rdcc_nbytes"], 256 * 1024 * 1024)
        self.assertEqual(kwargs["rdcc_nslots"], 20011)
        self.assertEqual(kwargs["rdcc_w0"], 0.75)
        self.assertEqual(kwargs["alignment_threshold"], 4096)
        self.assertEqual(kwargs["alignment_interval"], 4096)

    def test_open_kwargs_omits_page_buf_size_when_disabled(self) -> None:
        options = HDF5FileOptions()
        kwargs = options.open_kwargs(is_page_buffering_enabled=False)
        self.assertNotIn("page_buf_size", kwargs)
        self.assertNotIn("fs_strategy", kwargs)

    def test_open_kwargs_includes_page_buf_size_when_enabled(self) -> None:
        options = HDF5FileOptions()
        kwargs = options.open_kwargs(is_page_buffering_enabled=True)
        self.assertEqual(kwargs["page_buf_size"], options.page_buffer_size)

    def test_rejects_non_power_of_two_page_size(self) -> None:
        with self.assertRaises(ValueError):
            HDF5FileOptions(file_space_page_size=3000)

    def test_rejects_page_size_below_minimum(self) -> None:
        with self.assertRaises(ValueError):
            HDF5FileOptions(file_space_page_size=256)

    def test_rejects_page_buffer_not_multiple_of_page_size(self) -> None:
        with self.assertRaises(ValueError):
            HDF5FileOptions(file_space_page_size=4096, page_buffer_size=5000)

    def test_rejects_preemption_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            HDF5FileOptions(chunk_cache_preemption=1.5)


class TestHDF5LibraryVersion(unittest.TestCase):
    def test_bound_returns_value(self) -> None:
        self.assertEqual(HDF5LibraryVersion.LATEST.bound, "latest")
        self.assertEqual(HDF5LibraryVersion.EARLIEST.bound, "earliest")


class TestHDF5AccessMode(unittest.TestCase):
    def test_is_writable(self) -> None:
        self.assertTrue(HDF5AccessMode.APPEND.is_writable)
        self.assertFalse(HDF5AccessMode.READ_ONLY.is_writable)

    def test_h5py_mode(self) -> None:
        self.assertEqual(HDF5AccessMode.APPEND.h5py_mode, "a")
        self.assertEqual(HDF5AccessMode.READ_ONLY.h5py_mode, "r")


class TestHDF5FileSpaceStrategy(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(HDF5FileSpaceStrategy.PAGE.value, "page")
        self.assertEqual(HDF5FileSpaceStrategy.FSM_AGGR.value, "fsm")


if __name__ == "__main__":
    unittest.main()
