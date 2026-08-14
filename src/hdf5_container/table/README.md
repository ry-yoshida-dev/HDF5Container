# table

## Overview

A column-store layer for workloads that write many small, fixed-shape rows. Instead of one HDF5 group and
several datasets per row, a `ColumnTable` holds one resizable dataset per column shared by every row, with the
logical row count tracked separately from each column's allocated capacity. This turns what would be thousands
of small HDF5 objects into a handful of large, chunked datasets, which is what keeps per-row object-header and
allocation overhead low.

## Components

| Component | Description |
|-----------|-------------|
| [`column_spec.py`](./column_spec.py) | `ColumnSpec`, the static shape/dtype/chunking specification of one column. |
| [`growable_column.py`](./growable_column.py) | `GrowableColumn`, one resizable dataset with capacity growth, contiguous read/write, and unordered/duplicated fancy-index read/write. |
| [`column_table.py`](./column_table.py) | `ColumnTable`, a group of `GrowableColumn`s sharing one logical row count, with atomic-looking append semantics. |

## Examples

```python
import numpy as np

from hdf5_container import HDF5Container
from hdf5_container.table import ColumnSpec, ColumnTable

with HDF5Container.from_path("sample.h5") as container:
    group = container.access_subgroup(keys=["scene0", "rows"]).data
    specs = [
        ColumnSpec(name="box_coord", row_shape=(4,), dtype=np.dtype(np.float32)),
        ColumnSpec(name="box_conf", row_shape=(), dtype=np.dtype(np.float32)),
    ]
    table = ColumnTable.create(group=group, specs=specs, growth_quantum=8192)
    start = table.append(
        rows={
            "box_coord": np.zeros((3, 4), dtype=np.float32),
            "box_conf": np.zeros(3, dtype=np.float32),
        }
    )
    rows = table.read_slice(start=start, count=3)
```
