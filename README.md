## Neural Control: Adjoint Learning Through Equilibrium Constraints

This repository contains the C++ quasi-static rod simulator and the Python
learning scripts used in the manuscript *"Neural Control: Adjoint Learning
Through Equilibrium Constraints"*. The simulator is exposed to Python through
`pybind11` as the module `nn_der`, and the learning scripts in
`learning_scripts/` use it as a differentiable forward model for the three
control tasks reported in the paper.

---

### Repository layout

- `src/` &mdash; C++ quasi-static elastic-rod simulator (stretching, bending,
  twisting, gravity, damping, IMC contact, Newton solver with line search).
  `src/app.cpp` exposes the simulator to Python through `pybind11`.
- `nn_der/` &mdash; build output directory for the Python extension
  `nn_der.nn_der` (`nn_der*.so`).
- `learning_scripts/` &mdash; Python control scripts for the three tasks, one
  file per (task, method) pair.
- `learning_scripts/inputs/` &mdash; initial rod geometry and target shapes
  (`vertices*.txt`, `C_initial.txt`, `M_initial.txt`, `U_initial.txt`).
- `targets/` &mdash; target trajectories / shapes used by Tasks 2 and 3.
- `common.py`, `utils.py` &mdash; shared helpers (policy network, simulator
  reset, animation, thread configuration).
- `run_experiments.sh` &mdash; convenience launcher that runs a batch of
  experiments back-to-back.
- `experimental_results/`, `simulation_results/` &mdash; output directories
  populated by the learning scripts.

***

### How to use

#### 1. Build the C++ simulator binding

The simulator must be compiled before any learning script can run. The build
follows a standard CMake + `pip install -e .` flow and produces
`nn_der/nn_der*.so`, which the Python scripts import as `nn_der.nn_der`.

System dependencies (tested on Ubuntu 20.04&ndash;24.04 with Python 3.10+):

- Eigen 3.4.0
- Intel oneAPI MKL (Pardiso + BLAS/LAPACK backend for Eigen)
- SymEngine (built with `-DWITH_LLVM=on`)
- OpenGL / GLUT (`libglu1-mesa-dev freeglut3-dev mesa-common-dev`)
- pybind11 (`pip install pybind11`)
- Python packages: `torch`, `numpy`, `matplotlib`

Before configuring CMake, export the MKL root so that `find_package(MKL)`
succeeds (use whichever variable name your MKL version expects):

```bash
export MKLROOT=/opt/intel/oneapi/mkl/2022.0.2   # older versions
export MKL_DIR=/opt/intel/oneapi/mkl/2024.2     # newer versions
```

Then build and install the Python binding:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
pip install -e .   # installs the `nn_der` Python package
```

After a successful build, `python -c "import nn_der.nn_der"` should succeed
from the repository root.

#### 2. Run a single control experiment

Each script in `learning_scripts/` is self-contained and configures itself
through a top-level `CONFIG` dict at the top of the file (cases, horizon `T`,
learning rate, optimizer hyperparameters, etc.). To run a single experiment,
launch the corresponding script from the repository root, for example:

```bash
# Limit BLAS / MKL threads for stable per-iteration timings.
export OMP_NUM_THREADS=1

# Task 1 (any-node reaching) with the proposed Adjoint + RHC method.
python3 learning_scripts/any_node_adjoint_RHC.py

# Task 2 (middle-node trajectory tracking) with the baseline MPC.
python3 learning_scripts/middle_tracking_MPC.py

# Task 3 (shape control toward a letter target) with iCEM.
python3 learning_scripts/letter_curve_icem.py
```

The scripts write logs, learned policies, and rollouts under
`experimental_results/` and `simulation_results/`.

#### 3. Tasks and methods

The naming convention is `<task>_<method>.py`:

| Task prefix              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `any_node_*`             | Task 1 &mdash; drive a selected node of the elastic strip to a target.      |
| `middle_tracking_*`      | Task 2 &mdash; trace the middle node along a prescribed trajectory.         |
| `letter_curve_*`         | Task 3 &mdash; shape control toward a prescribed letter-shaped target.      |

| Method suffix            | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `*_adjoint_RHC.py`       | Proposed method: adjoint learning with receding-horizon control.            |
| `*_MPC.py`               | Adjoint-based MPC baseline (re-plans at every step, no policy).             |
| `*_noMPC.py`             | Open-loop adjoint optimization without receding-horizon control.            |
| `*_cem.py`               | Derivative-free baseline: CEM.                                              |
| `*_icem.py`              | Derivative-free baseline: iCEM.                                             |
| `*_spsa.py`              | Derivative-free baseline: SPSA.                                             |

Any of the nine (task, method) combinations above can be launched directly.

#### 4. Batch runs

`run_experiments.sh` chains several scripts back-to-back. Edit the
uncommented block at the bottom of the file to choose which experiments to
run, then:

```bash
bash run_experiments.sh
```

#### 5. Reproducing the paper&rsquo;s figures

The three tasks reported in the manuscript correspond to the three task
prefixes above. To reproduce the main comparison, run, for each task, the
`adjoint_RHC`, `MPC`, `noMPC`, `cem`, `icem`, and `spsa` variants and collect
the loss / wall-clock numbers from the per-script logs.

***

### TODO

#### High priority
- [ ] Provide a minimal `requirements.txt` / `pyproject.toml` for the Python
      side so that a fresh environment can install everything in one step.
- [ ] Add a Dockerfile that pre-installs MKL, SymEngine, Eigen and builds
      `nn_der` automatically.
- [ ] Provide a single-entry-point CLI (`python -m neural_control --task ...
      --method ...`) instead of one script per (task, method) pair.
- [ ] Document every key in the per-script `CONFIG` dict (units, valid
      ranges, effect on convergence).

#### Medium priority
- [ ] Extend the simulator binding to 3D rods (currently the experiments are
      run with `enable_2d_sim = true`).
- [ ] Expose the contact / friction parameters as Python-side knobs instead
      of compile-time defaults.
- [ ] Add unit tests for the adjoint gradient (finite-difference check
      against the C++ Jacobian).
- [ ] Add a deterministic-seed flag at the top of every learning script so
      that all baselines share the same RNG protocol.

#### Low priority
- [ ] Replace the OpenGL/GLUT viewer with an optional Magnum-based renderer
      for higher-quality figures.
- [ ] Add a `learning_scripts/configs/` directory of YAML files so the
      `CONFIG` dicts can be version-controlled separately from the code.
- [ ] Add a notebook walk-through that loads a saved policy and replays it
      against the simulator.

***

### Completed

- [x] C++ quasi-static rod simulator with stretching, bending, twisting,
      gravity, damping, and IMC contact forces (`src/`).
- [x] `pybind11` binding exposing the simulator as `nn_der.nn_der`
      (`src/app.cpp`, `CMakeLists.txt`, `setup.py`).
- [x] Adjoint + Receding-Horizon-Control implementation for all three tasks
      (`*_adjoint_RHC.py`).
- [x] Adjoint-based MPC and open-loop adjoint baselines
      (`*_MPC.py`, `*_noMPC.py`).
- [x] Derivative-free baselines: CEM, iCEM, SPSA
      (`*_cem.py`, `*_icem.py`, `*_spsa.py`) on all three tasks.
- [x] Validation on a learned DEQ-style equilibrium model trained from
      real slinky force&ndash;strain data (materials linked in the rebuttal
      supplement).
- [x] Quantitative comparison of time / memory complexity and best loss
      across all three tasks (see manuscript Table and supplement).
- [x] Original task videos and rollouts under `experimental_results/`.
