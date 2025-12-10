# GitHub Copilot Instructions for Visisipy

## Repository Overview

Visisipy (VISion Simulations In PYthon) is a Python library for optical simulations of the eye. It provides an easy-to-use interface to define and build eye models and perform common ophthalmic analyses on these models with support for multiple backends (OpticStudio and Optiland).

## Development Environment

### Python Version
- Minimum: Python 3.10
- Maximum: Python 3.13 (inclusive, <3.14)
- Default development version: Python 3.12

### Package Manager
- Primary: `uv` (recommended)
- Fallback: `pip`
- Use `hatch` for environment management and task execution

## Code Style and Formatting

### Linting and Formatting
- **Linter/Formatter**: Ruff (version 0.11.5)
- **Configuration**: `ruff_defaults.toml` with project-specific overrides in `pyproject.toml`
- **Line length**: 120 characters
- **Commands**:
  - Format code: `uvx hatch fmt`
  - Check formatting: `uvx hatch fmt --check`
  - Format docstrings: `uvx hatch run format-docstrings`

### Import Conventions
- **REQUIRED**: All files MUST include `from __future__ import annotations` as the first import
- Use absolute imports (relative imports are banned)
- Organize imports with known first-party package: `visisipy`

### Type Annotations
- Use type annotations for all public functions and methods
- Use `TYPE_CHECKING` guard for type-only imports to avoid circular dependencies
- Prefer modern type hints (e.g., `list[str]` over `List[str]`)

### Docstring Style
- **Format**: NumPy style (numpydoc)
- **Max line length**: 120 characters for docstrings
- **Max summary lines**: 1
- **Code in docstrings**: Format with max 80 characters per line
- All public APIs must have comprehensive docstrings

## Testing

### Test Framework
- **Framework**: pytest
- **Test Location**: `tests/` directory
- **Configuration**: See `[tool.pytest.ini_options]` in `pyproject.toml`

### Running Tests
```bash
# Run tests with numpy backend and no OpticStudio (default)
uvx hatch test --no-opticstudio

# Run tests with torch-cpu backend
uvx hatch run test-torch

# Run tests with torch-gpu backend
uvx hatch run test-gpu

# Run specific test file
uvx hatch test tests/test_specific.py --no-opticstudio
```

### Test Markers
- `needs_opticstudio`: Test requires OpticStudio and will be skipped if unavailable
- `windows_only`: Test can only run on Windows

### Test Coverage
- Omit `tests/*` from coverage reports
- Exclude `if TYPE_CHECKING:` and `raise NotImplementedError` from coverage

## Backend-Specific Details

### OpticStudio Backend
- **Platform**: Windows only
- **Dependency**: `zospy>=2.1.2` (only installed on Windows)
- Use platform checks: `if platform.system() == "Windows":`
- Tests in `tests/opticstudio/` are automatically skipped on non-Windows platforms

### Optiland Backend
- **Platform**: Cross-platform
- **Dependency**: `optiland>=0.5.8`
- **Computation backends**: numpy (default), torch
  - When using the torch backend, it can be configured to use the CPU or GPU
- More permissive for private attribute access (see `SLF001` rule exceptions)

## Project Structure

### Main Package (`visisipy/`)
- `analysis/`: Analysis functions (raytrace, etc.)
- `models/`: Eye model definitions (geometry, materials)
- `opticstudio/`: OpticStudio backend implementation (Windows only)
- `optiland/`: Optiland backend implementation
- `backend.py`: Backend management and abstraction
- `plots.py`: Visualization functions
- `refraction.py`: Refraction utilities
- `wavefront.py`: Wavefront utilities
- `types.py`: Type definitions

### Documentation
- **Tool**: Sphinx with ReadTheDocs
- **Location**: `docs/`
- **Config**: `.readthedocs.yml`
- **Build**: `hatch run docs:build`
- **Preview**: `hatch run docs:preview` (this will watch the documentation and rebuild on changes)

## Build System

### Build Backend
- **Tool**: Hatchling
- **Version management**: versioningit (dynamic versioning)
- **Configuration**: `pyproject.toml`

### Dependencies
- Core scientific stack: numpy, scipy, matplotlib, numba
- Optics: optiland (required), zospy (Windows only)
- Type hints: typing-extensions>=4.12.2 (automatically installed for Python <3.11)

## Coding Conventions

### Error Handling
- Allowed to use string literals and f-strings in exceptions in existing code (`EM101`, `EM102` ignored), but do not use string literals and f-strings in exceptions when writing new code
- Use specific exception types

### Assertions
- Do NOT use assertions in production code (except in `scripts/`)
- Use proper error handling with exceptions

### File Ignores
- `docs/*` and `examples/*`: Ignore `I002`, `INP001`, `T20`
- `scripts/*`: Allow assertions (`S101`)
- `tests/*`: More permissive rules for args, booleans, and pylint rules

### Naming Conventions
- Follow PEP 8 naming conventions
- Constants: UPPER_CASE
- Variables/functions: snake_case
- Classes: PascalCase
- Type variables: PascalCase

## Special Notes

### Installation
- Available on PyPI: `pip install visisipy`
- Available on Conda Forge: `conda install -c conda-forge visisipy`

### Citation
- Project has a `CITATION.cff` file
- DOI available via Zenodo

### Contributing
- See `CONTRIBUTING.md` and full guidelines in `docs/contributing.md`

## Common Commands

```bash
# Install dependencies
uv sync

# Format code
uvx hatch fmt

# Check formatting
uvx hatch fmt --check

# Format docstrings
uvx hatch run format-docstrings

# Run tests
uvx hatch test --no-opticstudio

# Build documentation
hatch run docs:build

# Preview documentation
hatch run docs:preview

# Build package
uv build
```
