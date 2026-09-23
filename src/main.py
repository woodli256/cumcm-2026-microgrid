# -*- coding: utf-8 -*-
"""论文配套源码统一入口。

环境 Python 3.13；安装依赖
python -m pip install numpy==2.5.1 scipy==1.18.0 openpyxl==3.1.5 matplotlib==3.10.8 Pillow==12.3.0 PyMuPDF==1.27.2.2

生成五份表格
python main.py --attachments "题目附件目录" --output-dir "重算表格目录"
如需覆盖已有表格，添加 --overwrite。

准备完整实验环境并重算主结果
python main.py --attachments "题目附件目录" --stage core --work-dir "外部计算目录"
执行全部补充实验和绘图（计算量较大）
python main.py --attachments "题目附件目录" --stage all --work-dir "外部计算目录"
也可先用 --stage prepare 准备环境，再在外部计算目录中单独运行 code 下的程序。
完整的执行顺序见下面的 EXPERIMENTS 常量。图表在补充实验完成后生成。

附件目录含附件1.xlsx至附件4.xlsx，以及附件5内的五份空模板。
原始附件只读。补充实验使用提交文件夹以外的专用计算目录，生成数据、
图片和检查记录均留在计算目录。代码目录只保留源码。

code 包含预测、调度、导出、敏感性、控制对照、跨日和电价实验及核验；
figures 包含论文数据图的绘制程序；_utils 包含绘图公共模块。
源码保留论文工程中的程序名称。旧分位数实验单独输出，不替代正式结果。
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NAMES = ('result1', 'result2', 'result3', 'result4-2', 'result4-3')
EXPERIMENTS = (
    ('sensitivity.py',),
    ('model_depth.py',),
    ('degradation_study.py',),
    ('forecast_uncertainty.py',),
    ('review_benchmarks.py',),
    ('improve_experiments.py',),
    ('improve_experiments.py', '--refine'),
    ('settlement_sensitivity.py',),
    ('risk_mpc.py', '--phase', 'train'),
    ('run_control_comparison.py',),
    ('check_risk_mpc.py', '--require-complete'),
    ('revision_experiments.py',),
    ('check_crossday_attribution.py',),
    ('run_q4_matched_controls.py',),
    ('analyze_q4_price_effects.py',),
    ('reproduce_legacy_price.py',),
    ('robust_price.py', '--run'),
    ('check_robust_price.py',),
    ('check_robust_price.py', '--table-only'),
    ('check_forecast_margin.py',),
    ('check_degradation_extension.py',),
    ('check_model_formulation.py',),
)


def run(script, *arguments, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    if cwd:
        env['MPLCONFIGDIR'] = str(cwd / '.mplconfig')
    print(f'运行 {script.name} {" ".join(map(str, arguments))}', flush=True)
    subprocess.run([sys.executable, '-B', str(script), *map(str, arguments)],
                   cwd=cwd, env=env, check=True)


def prepare(attachments, work, parser):
    package = HERE.parent
    if work == package or package in work.parents or work in HERE.parents:
        parser.error('--work-dir 必须位于提交文件夹外，且不能是提交文件夹的上级目录')
    marker = work / '.cumcm_source_workspace'
    if work.exists() and any(work.iterdir()) and not marker.is_file():
        parser.error('--work-dir 请选择空目录，或由本程序建立的专用计算目录')
    work.mkdir(parents=True, exist_ok=True)
    marker.write_text('C题论文源码计算目录\n', encoding='utf-8')
    for folder in ('code', 'figures', '_utils'):
        for source in (HERE / folder).rglob('*.py'):
            target = work / folder / source.relative_to(HERE / folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for folder in ('data/templates', 'results', 'tables', 'audit', 'figures/source_data'):
        (work / folder).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(attachments / '附件5' / f'{name}.xlsx',
                     work / 'data/templates' / f'{name}.xlsx')
    run(work / 'code/ingest.py', attachments, cwd=work)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attachments', required=True, type=Path)
    parser.add_argument('--stage', choices=('tables', 'prepare', 'core', 'all'), default='tables')
    parser.add_argument('--output-dir', type=Path, default=HERE.parent / '表格')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    attachments = args.attachments.resolve()
    required = [attachments / f'附件{i}.xlsx' for i in range(1, 5)]
    required += [attachments / '附件5' / f'{n}.xlsx' for n in NAMES]
    for path in required:
        if not path.is_file():
            parser.error(f'缺少附件 {path}')
    if args.stage == 'tables':
        flags = ['--attachments', attachments, '--output-dir', args.output_dir.resolve()]
        if args.overwrite:
            flags.append('--overwrite')
        run(HERE / 'code/export_five.py', *flags)
        return
    if args.work_dir is None:
        parser.error('prepare、core、all 阶段必须指定 --work-dir')
    work = args.work_dir.resolve()
    prepare(attachments, work, parser)
    if args.stage == 'prepare':
        return
    for filename in ('solve.py', 'validate.py', 'report_data.py'):
        run(work / 'code' / filename, cwd=work)
    run(work / 'code/export_xlsx.py', '--output-dir', work / 'results', '--overwrite', cwd=work)
    run(work / 'code/check_exports.py', cwd=work)
    if args.stage == 'all':
        for filename, *flags in EXPERIMENTS:
            run(work / 'code' / filename, *flags, cwd=work)
        for script in sorted((work / 'figures').glob('gen_fig_*.py')):
            run(script, cwd=work)
        run(work / 'code/revision_assets.py', cwd=work)
    print(f'计算完成，输出位于 {work}', flush=True)


if __name__ == '__main__':
    main()
