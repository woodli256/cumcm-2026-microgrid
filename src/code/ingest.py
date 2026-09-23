# -*- coding: utf-8 -*-
"""将题目附件转换为计算输入。原始Excel只读。"""
from pathlib import Path
import argparse,numpy as np
from openpyxl import load_workbook

def main():
    ap=argparse.ArgumentParser();ap.add_argument('attachments',type=Path)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'data/inputs.npz');args=ap.parse_args()
    a={}
    for filename in ['附件1.xlsx','附件2.xlsx','附件3.xlsx','附件4.xlsx']:
        w=load_workbook(args.attachments/filename,read_only=True,data_only=True)
        for s in w:
            rows=list(s.values);skip=2 if filename=='附件3.xlsx' else 1
            key={'附件1.xlsx':'base','附件3.xlsx':'forecast','附件4.xlsx':'price'}.get(filename,'load' if s.title=='小区负载' else 'pv')
            a[key]=np.array([row[skip:] for row in rows[1:]],float)
            assert np.isfinite(a[key]).all()
    assert a['load'].shape==a['pv'].shape==a['price'].shape==(365,144)
    assert a['base'].shape==(144,3) and a['forecast'].shape==(1460,24)
    args.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out,**a)
if __name__=='__main__':main()
