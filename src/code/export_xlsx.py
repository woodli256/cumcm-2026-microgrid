# -*- coding: utf-8 -*-
"""根据题目空模板和已验证JSON导出五份结果工作簿，仅依赖Python/openpyxl。

示例：python code/export_xlsx.py --output-dir .work/export_test
默认拒绝覆盖已有文件；明确需要重建时添加 --overwrite。
总量与费用写入JSON中未经舍入的已计算值，六位小数仅为显示格式。
"""
from pathlib import Path
from copy import copy
from datetime import datetime
import argparse
import json
import re
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
NUMBER_FORMAT = '0.000000'
DATE_FORMAT = 'yyyy-mm-dd'

def time_label(index):
    return f'{index // 6}:{index % 6 * 10:02d}'

def excel_value(value, column):
    if column == 1 and isinstance(value, str) and re.fullmatch(r'2025-\d{2}-\d{2}', value):
        return datetime.fromisoformat(value)
    return value

def export_workbooks(template_dir, output_dir, values_file, overwrite=False):
    payload = json.loads(Path(values_file).read_text(encoding='utf-8'))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check all destinations before writing any workbook.
    for name in payload:
        target = output_dir / f'{name}.xlsx'
        if target.exists() and not overwrite:
            raise FileExistsError(f'{target}已存在；如需重建，请使用--overwrite。')
    records = {}
    for name, sheets in payload.items():
        workbook = load_workbook(Path(template_dir) / f'{name}.xlsx')
        if set(workbook.sheetnames) != set(sheets):
            raise ValueError(f'{name}的模板工作表与JSON不一致')
        for sheet_name, rows in sheets.items():
            sheet = workbook[sheet_name]
            columns = len(rows[0])
            if not rows or any(len(row) != columns for row in rows):
                raise ValueError(f'{name}/{sheet_name}数据行不规则')
            old_last_row = sheet.max_row
            old_last_col = sheet.max_column
            # Keep template headers/styles. Sparse example rows become full chronological data.
            for merged in list(sheet.merged_cells.ranges):
                sheet.unmerge_cells(str(merged))
            row_style = [copy(sheet.cell(2, j)._style) for j in range(1, columns + 1)]
            row_align = [copy(sheet.cell(2, j).alignment) for j in range(1, columns + 1)]
            for row in sheet.iter_rows(min_row=2, max_row=max(old_last_row, len(rows)+1), max_col=max(old_last_col, columns)):
                for cell in row:
                    cell.value = None
            for i, row in enumerate(rows, 2):
                for j, value in enumerate(row, 1):
                    cell = sheet.cell(i, j)
                    if i > old_last_row:
                        cell._style = copy(row_style[j-1])
                        cell.alignment = copy(row_align[j-1])
                    # 沿用论文的结束时刻解释，时间区间由 report_data.py 生成。
                    cell.value = excel_value(value, j)
            interval_matrix = name != 'result1' and sheet_name in ('计划购电量', '调整购电量')
            end_row = len(rows) + 1
            if interval_matrix:
                for col in range(2, 146):
                    k = col - 2
                    sheet.cell(1, col).value = f'{time_label(k)}-{time_label(k+1)}'
                if sheet_name == '调整购电量':
                    sheet['EQ1'] = '全天总费用'
                sheet.column_dimensions['A'].width = 14
                for col in range(2, 148):
                    sheet.column_dimensions[get_column_letter(col)].width = 16
                for i in range(2, end_row+1):
                    sheet.cell(i, 1).number_format = DATE_FORMAT
                    for col in range(2, 148):
                        sheet.cell(i, col).number_format = NUMBER_FORMAT
                sheet.freeze_panes = 'B2'
            elif name == 'result1':
                sheet.column_dimensions['A'].width = 19
                for col in range(2, columns+1):
                    sheet.column_dimensions[get_column_letter(col)].width = 18
                numeric_cols = (2,) if sheet_name == '计划购电量' else (2, 3, 5)
                for i in range(2, end_row+1):
                    sheet.row_dimensions[i].height = 22
                    for col in numeric_cols:
                        sheet.cell(i, col).number_format = NUMBER_FORMAT
            else:
                sheet.column_dimensions['A'].width = 14
                sheet.column_dimensions['B'].width = 21
                for col in range(3, columns+1):
                    sheet.column_dimensions[get_column_letter(col)].width = 18
                for i in range(2, end_row+1):
                    sheet.row_dimensions[i].height = 22
                    sheet.cell(i, 1).number_format = DATE_FORMAT
                    for col in range(3, columns+1):
                        sheet.cell(i, col).number_format = NUMBER_FORMAT
            sheet.row_dimensions[1].height = 28
        target = output_dir / f'{name}.xlsx'
        workbook.save(target)
        workbook.close()
        records[name] = {'file': str(target), 'sheets': len(sheets), 'data_cells': sum(len(row) for rows in sheets.values() for row in rows)}
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-dir', type=Path, default=ROOT/'data/templates')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'提交表格')
    parser.add_argument('--values-file', type=Path, default=ROOT/'results/export_values.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    records = export_workbooks(args.template_dir, args.output_dir, args.values_file, args.overwrite)
    print(json.dumps(records, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
