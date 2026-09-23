# -*- coding: utf-8 -*-
"""对同一因果策略在减购退款口径下顺序重算两种日内合同场景。"""
from pathlib import Path
import json
import numpy as np
from solve import ROOT, read_inputs, forecasts, run, totals


def main():
    inputs = read_inputs(ROOT / 'data/inputs.npz')
    fc = forecasts(inputs)
    answer = {}
    for key, dynamic in [('q3', False), ('q4_3', True)]:
        o = run(inputs, fc, use=True, dynamic=dynamic, q=.65, refund=True)
        np.savez_compressed(ROOT / 'results' / f'refund_{key}.npz', **o)
        answer[key] = totals(o)
        print(key, answer[key], flush=True)
    (ROOT / 'results' / 'settlement_sensitivity.json').write_text(
        json.dumps(answer, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
