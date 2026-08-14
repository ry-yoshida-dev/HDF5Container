# mixin

## Overview

Mixin classes that compose reusable behavior for `HDF5Container`.
They separate I/O operations, special methods, and protocol typing from the main container logic.

## Components

| Component | Description |
|-----------|-------------|
| [`io.py`](./io.py) | File-backed helpers: `from_path` (create/open with `HDF5AccessMode`/`HDF5FileOptions`), `reopen` (promote/demote access mode on an existing file-backed container), `flush`, and `close`. |
| [`special_methods.py`](./special_methods.py) | Context manager (`__enter__`, `__exit__`) and iterator (`__iter__`) behavior. |
| [`protocols.py`](./protocols.py) | Protocol contracts used to type mixin/container interactions. |
