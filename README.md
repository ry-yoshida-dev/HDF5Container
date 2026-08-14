# HDF5Container

## Overview

HDF5Container (`hdf5_container`) is a lightweight Python package for hierarchical HDF5 read/write operations.
It provides a simple interface for storing values into nested groups, retrieving datasets, and handling periodic flushes.
It also provides a column-store layer (`hdf5_container.table`) for workloads that need many small rows without
paying HDF5's per-object overhead, backed by tunable file creation/open options (`hdf5_container.options`).

For module-level details, see [src/hdf5_container/README.md](src/hdf5_container/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Dependencies are installed automatically.
To install dependencies only:

```bash
pip install -r requirements.txt
```

## Example

```python
from hdf5_container import HDF5Container

with HDF5Container.from_path("sample.h5") as container:
    container.store(keys=["users", "alice"], name="age", data=30)
    container.store(keys=["users", "alice"], name="city", data="Tokyo")
    age = container.access_value(keys=["users", "alice"], name="age")
    city = container.access_value(keys=["users", "alice"], name="city")
    print(age, city)  # 30 Tokyo
```

### Column store example

```python
import numpy as np

from hdf5_container import ColumnSpec, ColumnTable, HDF5Container

with HDF5Container.from_path("sample.h5") as container:
    group = container.access_subgroup(keys=["scene0", "rows"]).data
    specs = [
        ColumnSpec(name="box_coord", row_shape=(4,), dtype=np.dtype(np.float32)),
        ColumnSpec(name="box_conf", row_shape=(), dtype=np.dtype(np.float32)),
    ]
    table = ColumnTable.create(group=group, specs=specs)
    table.append(
        rows={
            "box_coord": np.zeros((3, 4), dtype=np.float32),
            "box_conf": np.zeros(3, dtype=np.float32),
        }
    )
```
