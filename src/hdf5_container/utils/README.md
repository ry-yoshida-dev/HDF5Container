# utils

## Overview

Small standalone helpers used by `HDF5Container` that do not belong to the container class itself.

## Components

| Component | Description |
|-----------|-------------|
| [`counter.py`](./counter.py) | `FlushCounter`, a shared mutable counter used to trigger periodic flushes. |
| [`reset.py`](./reset.py) | `reset_hdf5()`, recreates an HDF5 file from scratch and returns a container for it. |
