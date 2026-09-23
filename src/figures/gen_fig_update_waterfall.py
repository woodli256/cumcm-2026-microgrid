"""当前论文进阶组合图，可独立重绘。"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _utils.advanced_paper_figures import update_waterfall
update_waterfall()
