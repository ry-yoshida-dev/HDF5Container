# options

## Overview

Enumerations and a tuning-options dataclass that control how `HDF5Container` creates and (re)opens HDF5 files:
library version bound, file space management strategy, chunk/page cache sizing, and access mode. `HDF5FileOptions`
defaults are tuned for files holding many small objects (paged allocation, a large page buffer, and a large chunk
cache), which is what keeps per-row overhead low for the column-store layer in `hdf5_container.table`.

## Components

| Component | Description |
|-----------|-------------|
| [`library_version.py`](./library_version.py) | `HDF5LibraryVersion`, the library format version bound passed to h5py's `libver`. |
| [`file_space_strategy.py`](./file_space_strategy.py) | `HDF5FileSpaceStrategy`, the file space management strategy used at file creation. |
| [`access_mode.py`](./access_mode.py) | `HDF5AccessMode`, read-only vs. append file access. |
| [`file_options.py`](./file_options.py) | `HDF5FileOptions`, the creation/open tuning dataclass and its `creation_kwargs()`/`open_kwargs()` builders. |
