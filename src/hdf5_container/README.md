# hdf5_container

## Overview

Small HDF5 utility package built on top of `h5py`.
It provides a container class for hierarchical group access, dataset read/write, and periodic flush handling,
plus a column-store table layer for workloads that write many small, fixed-shape rows without paying HDF5's
per-object overhead of one group and several datasets per row.

## Components

| Component | Description |
|-----------|-------------|
| [`container.py`](./container.py) | Core `HDF5Container` implementation for store/access operations. |
| [`utils/`](./utils/README.md) | Utility helpers such as `reset_hdf5()` for recreating container files. |
| [`mixin/`](./mixin/README.md) | Mixin layer for I/O methods, special methods, and typing protocols. |
| [`options/`](./options/README.md) | File creation/open tuning options and their enums. |
| [`table/`](./table/README.md) | Column-store table layer (`ColumnSpec`, `GrowableColumn`, `ColumnTable`). |
