# Vendored: ana_cont

Upstream: <https://github.com/josefkaufmann/ana_cont>
Pinned at upstream commit `ab5aa80` (v1.1.2). MIT licence — see `LICENSE`.
Reference: J. Kaufmann and K. Held, *ana_cont: Python package for analytic
continuation*, Comput. Phys. Commun. **282**, 108519 (2023).

Vendored rather than declared as a dependency because the package is not on
PyPI, and a benchmark that only runs after a manual `git clone` is a benchmark
nobody runs.

**What is kept:** the four importable modules (`continuation.py`, `kernels.py`,
`solvers.py`, `__init__.py`) plus `LICENSE`. The GUI, packaging files, test
data, and the Padé Cython extension are not kept — `spevae.benchmark.maxent`
uses only the maximum-entropy solver, and `solvers.py` imports `pade` lazily
inside the Padé class, so its absence costs nothing.

**Local patch on top of upstream** (required for NumPy >= 2):

- `solvers.py`: `np.trapz` -> `np.trapezoid`, 7 call sites.

ana_cont uses relative imports throughout (`from . import kernels`), so it works
unmodified under this namespace. If you refresh the vendored copy, re-apply the
NumPy patch (or check whether upstream has adopted it) and update the pin above.

**Kernel convention.** ana_cont's `time_fermionic` kernel is
`exp(-tau*omega) / (1 + exp(-omega))` with `tau` rescaled to `[0,1]` and `omega`
rescaled by `beta` — i.e. *exactly* the dimensionless kernel this package uses
(`spevae.physics.kernel`), and exactly the one SmoQyDEAC uses. All three methods
therefore solve the same equation with the same sign and normalisation
conventions, which is what makes the three-way comparison meaningful. There is a
test asserting it.
