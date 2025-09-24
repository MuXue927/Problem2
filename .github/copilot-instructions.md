# AI Coding Agent Instructions for Multi-Period Product Distribution Optimization

## Project Overview
This codebase implements multiple algorithmic approaches (ALNS, Column Generation, Monolithic) for solving a multi-period product distribution problem using Gurobi optimization. The core problem involves assigning vehicles to transport products from plants to dealers across multiple time periods while managing inventory constraints.

## Architecture & Key Components

### Core Data Structures
- **`DataALNS`** (`InputDataALNS.py`): Central data container with sets/dicts for plants, dealers, SKUs, demands, production, inventory limits. Use `data.demands[(dealer, sku)]` for requirements, `data.sku_sizes[sku]` for volumes.
- **`SolutionState`** (`alnsopt.py`): Solution representation with `vehicles` list and `s_ikt` inventory dict. Always call `state.compute_inventory()` after modifications to update inventory levels.
- **`Vehicle`** (`vehicle.py`): Represents transport assignments with `cargo` dict `{(sku, day): qty}`, `capacity` tracking, and methods like `load()` for adding products.

### Algorithmic Approaches
- **ALNS** (`ALNSCode/`): Uses destroy/repair operators (e.g., `periodic_shaw_removal` in `alnsopt.py`) with acceptance criteria. Entry point: `main.py` with `ALNSOptimizer` class.
- **Column Generation** (`CGcode/`): Decomposes into master/sub problems. Master model in `masterModel.py`, subproblems in `subModel.py`.
- **Monolithic** (`monolithic/`): Direct Gurobi model in `MonolithicModelMultiple.py`.

### Data Flow
Input datasets (`datasets/multiple-periods/`) → Data loading → Initial solution generation → Optimization iterations → Output results (`OutPut-ALNS/`, `logs-alns/`)

## Critical Workflows

### Running ALNS Optimization
```bash
cd ALNSCode
python main.py  # Uses ALNSConfig for parameters
```
- Configures via `alns_config.py` (SEED, DATASET_TYPE, etc.)
- Outputs to `OutPut-ALNS/multiple-periods/{dataset}/opt_result.csv`
- Logs to `alns_optimization.log` and `logs-alns/`

### Debugging Feasibility
- Use `SolutionState.validate()` to check violations (negative inventory, overload, unmet demand)
- Inventory computed via `compute_inventory()` - essential after any cargo changes
- Vehicles have capacity constraints; use `Vehicle.load()` to add cargo safely

### Testing Changes
- Modify operators in `alnsopt.py`, test via `main.py` runs
- Check objective via `SolutionState.objective()` (includes penalties for infeasibility)
- Compare against baselines in `logs-alns/` or CG/monolithic outputs

## Project-Specific Conventions

### Naming & Patterns
- Inventory: `s_ikt[(plant, sku, day)]` for stock levels, `historical_s_ikt` for initial data
- Demands: `demands[(dealer, sku)]` total across periods
- Production: `sku_prod_each_day[(plant, sku, day)]` for daily output
- Operators: Functions like `random_removal(current, rng, degree=0.3)` return modified `SolutionState`
- Always copy states: `new_state = current.copy()` before modifications

### Error Handling
- Infeasible solutions return `float('inf')` from `objective()`
- Use `try/except` in data loading; log errors to `alns_optimization.log`
- Validate after changes: `feasible, violations = state.validate()`

### Dependencies
- Gurobi for optimization (models in `.lp` files, solved via `model.optimize()`)
- ALNS library (`from alns import ALNS`) for metaheuristic framework
- Numpy for features/clustering in operators like `periodic_shaw_removal`

### File Organization
- `ALNSCode/`: ALNS implementation
- `CGcode/`: Column generation approach  
- `monolithic/`: Direct optimization
- `datasets/`: Input data by type (e.g., `multiple-periods/`)
- `logs-*/`: Optimization logs by method
- `OutPut-*/`: Result CSVs by method

## Common Patterns & Examples

### Adding a New Operator
```python
def my_removal(current: SolutionState, rng, degree=0.3) -> SolutionState:
    state = current.copy()
    # ... removal logic ...
    state.compute_inventory()
    return state
```
Register in `main.py` via `alns.add_destroy_operator(my_removal, name="my_removal")`

### Checking Inventory After Changes
```python
veh.load(sku, qty)
state.compute_inventory()  # Updates s_ikt
if state.s_ikt[(plant, sku, day)] < 0:
    # Handle negative inventory
```

### Feature Engineering for Operators
Use numpy arrays for clustering (see `periodic_shaw_removal`):
```python
features = np.array([day, demand_avg, inv_level])
kmeans = KMeans(n_clusters=k, random_state=rng.integers(1000))
```

Focus on inventory balance and capacity constraints when modifying solutions.