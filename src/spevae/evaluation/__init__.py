r"""``spevae eval``: one frozen suite that every continuation method goes through.

- :mod:`.cases`   -- the frozen truths, noise draws and the real-data cell
- :mod:`.methods` -- methods behind one interface, and the shared scorer
- :mod:`.suite`   -- tiers, tasks, the parallel runner
- :mod:`.stats`   -- hierarchical bootstrap intervals, paired comparisons
- :mod:`.report`  -- scorecard.json, summary.md, figures
- :mod:`.acceptance` -- reproduce the banked numbers before judging anything new
"""

from .cases import CASES, draw, holstein_cell, realize
from .report import write_report
from .suite import SUITE_VERSION, TIERS, run_suite

__all__ = ["CASES", "realize", "draw", "holstein_cell", "run_suite", "TIERS",
           "SUITE_VERSION", "write_report"]
