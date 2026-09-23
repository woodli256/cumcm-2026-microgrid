# -*- coding: utf-8 -*-
"""按用户 Downloads/cumcm-paper-figures 的要求输出可编辑图件与数据。"""
from pathlib import Path
import json, sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
 sys.path.insert(0,str(ROOT))
from _utils import plot_style as PS
BLUE,ORANGE,GREEN,PURPLE,GRAY=PS.BLUE,PS.ORANGE,PS.GREEN,PS.PURPLE,PS.GRAY
BLUE_D,BLUE_L,RED,GRAY_L=PS.BLUE_D,PS.BLUE_L,PS.RED,PS.GRAY_L
INK=PS.INK
def paper_fig(w,h):
 """统一图宽（167.6 mm），高度按原比例换算，保持正文版式不变。"""
 return (PS.FIG_WIDTH_IN,round(h*PS.FIG_WIDTH_IN/float(w),3))
def cjk_font():
 return PS.cjk_font_path()
def setup():
 return PS.apply(9.2)
def clean(ax,axis='y'):
 return PS.clean(ax,grid=axis)
def save_bundle(fig,stem,arrays,metadata):
 serial={k:np.asarray(v) for k,v in arrays.items()}
 assert all(np.isfinite(v).all() for v in serial.values())
 meta=dict(metadata,data_status='real')
 meta.setdefault('sources',[])
 info=PS.save_bundle(fig,stem,arrays=serial,meta=meta)
 print(json.dumps({'figure':stem,'points':info.get('shapes',{}),'outputs':info['outputs']},ensure_ascii=False))
 return info
