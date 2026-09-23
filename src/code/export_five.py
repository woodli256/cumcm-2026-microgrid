# -*- coding: utf-8 -*-
"""从官方附件求解并生成五份结果表，无需随包提供原始数据或中间结果。

环境：Python 3.13，numpy==2.5.1，scipy==1.18.0，openpyxl==3.1.5。
安装：python -m pip install numpy==2.5.1 scipy==1.18.0 openpyxl==3.1.5
运行：python export_five.py --attachments "题目附件目录" --output-dir "重算表格目录"
附件目录须含附件1.xlsx至附件4.xlsx，以及附件5中的五份原始空模板。
默认输出到同级的表格目录。已有同名文件时拒绝覆盖，添加--overwrite可重建。

时间和费用口径沿用论文。原始0:10数据对应0:00-0:10区间；全天购电费
包括计划、紧急和适用的调整费用。计算过程文件使用系统临时目录并自动清理。
"""
from pathlib import Path
from datetime import date, timedelta
import argparse
import json
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import numpy as np
from openpyxl import load_workbook
from export_xlsx import export_workbooks

HERE = Path(__file__).resolve().parent
NAMES = ['result1', 'result2', 'result3', 'result4-2', 'result4-3']


def interval(start, end=None):
    def hm(t):
        return f'{t // 6}:{t % 6 * 10:02d}'
    return f'{hm(start)}-{hm(start + 1 if end is None else end)}'


def read_attachments(directory, target):
    data = {}
    for filename in ['附件1.xlsx', '附件2.xlsx', '附件3.xlsx', '附件4.xlsx']:
        workbook = load_workbook(directory / filename, read_only=True, data_only=True)
        for sheet in workbook:
            rows = list(sheet.values)
            key = {'附件1.xlsx': 'base', '附件3.xlsx': 'forecast',
                   '附件4.xlsx': 'price'}.get(filename, 'load' if sheet.title == '小区负载' else 'pv')
            skip = 2 if key == 'forecast' else 1
            data[key] = np.array([row[skip:] for row in rows[1:]], dtype=float)
            if not np.isfinite(data[key]).all():
                raise ValueError(f'{filename}/{sheet.title}含非数值或缺失项')
        workbook.close()
    if data['load'].shape != (365, 144) or data['pv'].shape != (365, 144) or data['price'].shape != (365, 144):
        raise ValueError('负载、光伏或电价数据维度不符')
    if data['base'].shape != (144, 3) or data['forecast'].shape != (1460, 24):
        raise ValueError('附件1或预报数据维度不符')
    np.savez_compressed(target, **data)


def emergency_events(values):
    events = []
    i = 0
    while i < 144:
        if values[i] <= 1e-6:
            i += 1
            continue
        start = i
        while i < 144 and values[i] > 1e-6:
            i += 1
        events.append([interval(start, i), float(values[start:i].sum())])
    return events or [['无', 0.0]]


def table_values(result_dir):
    exports = {}
    q1 = dict(np.load(result_dir / 'q1.npz'))
    exports['result1'] = {
        '计划购电量': [[interval(i), float(q1['g'][i])] for i in range(144)],
        '充放电量': [[interval(j, j+24), float(q1['c'][j:j+24].sum()),
                     float(q1['d'][j:j+24].sum()),
                     '0:00' if j == 0 else '24:00' if j == 24 else None,
                     float(q1['s'][0]) if j == 0 else float(q1['s'][-1]) if j == 24 else None]
                    for j in range(0, 144, 24)]}
    for key, filename in [('q2', 'result2'), ('q3', 'result3'),
                          ('q4_2', 'result4-2'), ('q4_3', 'result4-3')]:
        o = dict(np.load(result_dir / f'{key}.npz'))
        book = {'计划购电量': [], '充放电量': [], '紧急购电量': []}
        if key in ['q3', 'q4_3']:
            book['调整购电量'] = []
        for i in range(31, 365):
            day = (date(2025, 1, 1) + timedelta(days=i)).isoformat()
            total_cost = float(o['cost'][i].sum())
            book['计划购电量'].append([day, *o['g'][i].tolist(), float(o['g'][i].sum()), total_cost])
            if '调整购电量' in book:
                book['调整购电量'].append([day, *o['a'][i].tolist(), float(o['a'][i].sum()), total_cost])
            for j in range(0, 144, 24):
                book['充放电量'].append([
                    day if j == 0 else None, interval(j, j+24),
                    float(o['c'][i, j:j+24].sum()), float(o['d'][i, j:j+24].sum()),
                    '0:00' if j == 0 else '24:00' if j == 24 else None,
                    float(o['s'][i, 0]) if j == 0 else float(o['s'][i, -1]) if j == 24 else None])
            for j, event in enumerate(emergency_events(o['e'][i])):
                book['紧急购电量'].append([day if j == 0 else None, *event])
        exports[filename] = book
    return exports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attachments', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE.parents[1] / '表格')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    attachments = args.attachments.resolve()
    output = args.output_dir.resolve()
    templates = attachments / '附件5'
    required = [attachments / f'附件{i}.xlsx' for i in range(1, 5)]
    required += [templates / f'{name}.xlsx' for name in NAMES]
    for source in required:
        if not source.is_file():
            parser.error(f'缺少官方附件 {source}')
    for name in NAMES:
        if (output / f'{name}.xlsx').exists() and not args.overwrite:
            parser.error(f'输出已存在 {name}.xlsx，请选择新的输出目录或添加--overwrite')
    with tempfile.TemporaryDirectory(prefix='cumcm_compute_') as work:
        temp = Path(work)
        inputs = temp / 'inputs.npz'
        result_dir = temp / 'results'
        read_attachments(attachments, inputs)
        subprocess.run([sys.executable, '-B', str(HERE / 'solve.py'),
                        '--inputs', str(inputs), '--output', str(result_dir)], check=True)
        payload = temp / 'export_values.json'
        payload.write_text(json.dumps(table_values(result_dir), ensure_ascii=False), encoding='utf-8')
        export_workbooks(templates, output, payload, args.overwrite)
    print(f'五份结果表已保存到 {output}')


if __name__ == '__main__':
    main()
