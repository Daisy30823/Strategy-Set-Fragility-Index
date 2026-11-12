"""Regenerate all main and Extended Data figures from the stored results.
Usage (from the code directory):  python run_figures.py RESULTS_DIR"""
import os
import sys
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import apply_figure_style, panel_letter  # noqa: F401
spec = importlib.util.spec_from_file_location("claims", os.path.join(HERE, "claims.py"))
CL = importlib.util.module_from_spec(spec); spec.loader.exec_module(CL)
os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results"))
OUT = "."
src1 = open(os.path.join(HERE, "make_figs2.py"), encoding="utf-8").read().replace('OUT = "rev_out"', 'OUT = "."')
exec(src1)
exec(open(os.path.join(HERE, "make_ed2.py"), encoding="utf-8").read())
