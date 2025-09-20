## FINM 37000 - Futures and Related Derivates

This repo contains a Python package and course material for FINM 37000 offered in
Fall 2025 at the University of Chicago.

### The finm37000 package

Use of a `venv` is **strongly** encouraged. Different IDEs and systems will manage these differently.
Refer to documentation for your environment.

**DO NOT SKIP SETTING UP YOUR `venv`** The following directions assume you are working in a virtual
environment, but it is incumbent on you to make sure you are using yours because the instructions
make no attempt to cover the system and environment differences that can arise.

Install:
```
pip install 
```

Install for development:
```
git clone
cd finm37000-2025
python -m pip install --group dev -e .
python -m ruff check
python -m ruff format
python -m mypy .
```

Package source is available in the `src` directory with specifications in `pyproject.toml`.

### The FINM 3700 Course

The non-`src` directories contain other course materials for FINM 37000 - Futures and Related Derivatives.



