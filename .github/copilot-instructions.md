# Copilot Instructions for pytorch_study

This project is a collection of PyTorch and data processing scripts, each demonstrating a specific concept or workflow. There is no monolithic application structure; instead, each numbered Python file (1.py to 9.py) is a self-contained example. Data files are located in the `data/` directory.

## Project Structure
- `1.py` to `9.py`: Each file is a standalone script, typically focused on a single PyTorch or data manipulation topic (e.g., tensor operations, missing value handling, broadcasting, memory management).
- `data/`: Contains CSV files used by some scripts (e.g., `house_tiny.csv`, `sensor_data.csv`).

## Key Patterns & Conventions
- **Script Naming**: Scripts are named numerically in order of concept progression. There is no central entry point.
- **Data Loading**: Use absolute or relative paths to load CSVs from the `data/` directory. Example: `pd.read_csv('data/house_tiny.csv')` or with full path.
- **PyTorch Usage**: Most scripts use `torch` for tensor creation, manipulation, and computation. Examples include `torch.arange`, `torch.tensor`, `torch.sum`, and broadcasting.
- **Pandas Integration**: Some scripts use `pandas` for CSV reading and missing value handling before converting to tensors.
- **Memory Management**: Some scripts (e.g., `5.py`) intentionally test memory limits with large tensors and handle exceptions.
- **In-place vs. Out-of-place Operations**: Scripts (e.g., `2.py`) demonstrate the difference between in-place and out-of-place tensor operations, often printing object IDs to show memory effects.
- **Chinese Comments**: Many scripts include Chinese comments explaining the code and concepts.

## Developer Workflows
- **Run Scripts**: Execute any script directly: `python 3.py` (no build system or test runner is present).
- **Data Preparation**: Some scripts generate their own data files if not present (e.g., `3.py` writes `house_tiny.csv`).
- **Dependencies**: Requires `torch` and `pandas` (install via `pip install torch pandas`).

## Examples
- Handling missing values: See `3.py` and `4.py` for pandas fillna and column dropping.
- Tensor operations: See `1.py`, `6.py`, `7.py`, `8.py`, `9.py` for sum, cumsum, broadcasting, and arithmetic.
- Memory error handling: See `5.py` for try/except around large tensor creation.

## Integration Points
- No external APIs or services are used. All data is local.
- No custom modules or packages; all logic is in top-level scripts.

## Recommendations for AI Agents
- Treat each script as independent unless otherwise indicated.
- When adding new examples, follow the numeric naming and keep each script focused and self-contained.
- Use the `data/` directory for any new data files.
- Maintain Chinese comments for consistency if adding new scripts.

---
For questions or unclear conventions, review the most similar numbered script for reference.
