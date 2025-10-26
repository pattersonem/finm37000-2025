## FINM 37000 - Futures and Related Derivates

This repo contains a Python package and course material for FINM 37000 offered in
Fall 2025 at the University of Chicago.

### The finm37000 package

Use of a `venv` is **strongly** encouraged. Different IDEs and systems will manage these differently.
Refer to documentation for your environment.

**DO NOT SKIP SETTING UP YOUR `venv`** The following directions assume you are working in a virtual
environment, but it is incumbent on you to make sure you are using yours because the instructions
make no attempt to cover the system and environment differences that can arise.

You can install the package directly from `github`, but you will not get the other
course materials like notebooks and homework, so probably proceed to clone or fork and clone.

```
python -m pip install git+https://github.com/pattersonem/finm37000-2025.git
```

Install for development:
```
git clone git@github.com:pattersonem/finm37000-2025.git
cd finm37000-2025
# pip >= 25.2
python -m pip install --group demo -e .
# Earlier pip
python -m pip install -e ".[demo]"
```

### Test

```
python -m pytest
```

### Lint

```
python -m ruff check
```

### Format

```
python -m ruff format
```

### Type Check

```
python -m mypy .
```

Package source is available in the `src` directory with specifications in `pyproject.toml`.

### The FINM 3700 Course

The non-`src` directories contain other course materials for FINM 37000 - Futures and Related Derivatives.



