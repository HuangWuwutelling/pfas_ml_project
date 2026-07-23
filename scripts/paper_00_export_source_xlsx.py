"""
Export all sheets from es4c13284_si_002.xlsx to CSV files.
Uses only Python stdlib (zipfile + xml.etree.ElementTree).

Output:
  - PFAS_Properties.csv  (66 rows, PFAS compounds with SMILES)
  - All_data.csv         (1276 rows, raw data before cleaning)
  - Outlier_ID.csv       (57 rows, removed outliers)
  - Final_data.csv       (1851 rows, cleaned dataset with 67 columns)
  - SoilGrids_KNN.csv    (2040 rows, soil property interpolation)
  - KdvsOC_notes.csv     (empty sheet, only notes)
"""

import zipfile
import xml.etree.ElementTree as ET
import csv
import os
import re

INPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'source', 'es4c13284_si_002.xlsx')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'paper')

# Namespace
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

# --- Helper: build sheet name -> file number map ---
def build_sheet_map(zf):
    wb_tree = ET.fromstring(zf.read('xl/workbook.xml'))
    sheet_map = {}
    for i, sh in enumerate(wb_tree.findall('.//s:sheet', NS), 1):
        sheet_map[sh.get('name')] = i
    return sheet_map

# --- Helper: parse shared strings ---
def parse_shared_strings(zf):
    ss_tree = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    strings = []
    for si in ss_tree.findall('.//s:si', NS):
        t_elem = si.find('.//s:t', NS)
        strings.append(t_elem.text if t_elem is not None else '')
    return strings

# --- Helper: get cell value ---
def get_cell_value(cell, shared_strings):
    v = cell.find('s:v', NS)
    t = cell.get('t', '')
    if v is not None and v.text:
        if t == 's':
            idx = int(v.text)
            return shared_strings[idx] if idx < len(shared_strings) else f'?{idx}'
        return v.text
    return ''

# --- Helper: extract all data from a sheet as list of lists ---
def extract_sheet(zf, sheet_id, shared_strings):
    fname = f'xl/worksheets/sheet{sheet_id}.xml'
    if fname not in zf.namelist():
        return []
    tree = ET.fromstring(zf.read(fname))
    rows = tree.findall('.//s:row', NS)
    data = []
    for row in rows:
        cells = row.findall('s:c', NS)
        vals = [get_cell_value(c, shared_strings) for c in cells]
        # Pad to consistent width
        data.append(vals)
    return data

# --- Helper: pad all rows to same width ---
def pad_rows(data):
    if not data:
        return data
    max_cols = max(len(r) for r in data)
    return [r + [''] * (max_cols - len(r)) for r in data]

# --- Helper: combine two-header rows ---
def combine_headers(data):
    """Combine first two rows into single header: 'Col0_Row0 | Col1_Row1'"""
    if len(data) < 2:
        return data
    h1 = data[0]
    h2 = data[1]
    combined = []
    for i in range(max(len(h1), len(h2))):
        a = h1[i].strip() if i < len(h1) else ''
        b = h2[i].strip() if i < len(h2) else ''
        if a and b:
            combined.append(f'{a} ({b})')
        elif a:
            combined.append(a)
        elif b:
            combined.append(b)
        else:
            combined.append('')
    return [combined] + data[2:]

# --- Main ---
def main():
    zf = zipfile.ZipFile(INPUT)
    shared_strings = parse_shared_strings(zf)
    sheet_map = build_sheet_map(zf)

    # Define sheets to export: (output_name, sheet_name, combine_two_header_rows)
    targets = [
        ('PFAS_Properties', 'PFAS Properties', False),  # different layout, just raw
        ('All_data',         'All data',         True),
        ('KdvsOC_notes',     'KdvsOC',           False),
        ('Outlier_ID',       'Outlier ID',       True),
        ('Final_data',       'Final data',       True),
        ('SoilGrids_KNN',    'SoilGrids (KNN)',  False),  # single header row
    ]

    for out_name, sheet_name, combine_two in targets:
        sid = sheet_map.get(sheet_name)
        if not sid:
            print(f'[SKIP] {sheet_name}: not found')
            continue

        data = extract_sheet(zf, sid, shared_strings)
        if not data:
            print(f'[SKIP] {sheet_name}: empty')
            continue

        data = pad_rows(data)

        # For PFAS Properties, it has multi-row header, just take rows >= 8
        if sheet_name == 'PFAS Properties':
            # Find where column header starts (row with "PFAS abbreviation" etc.)
            header_idx = None
            for i, row in enumerate(data):
                vals = [c.strip().lower() for c in row[:5]]
                if 'pfas abbreviation' in ' '.join(vals):
                    header_idx = i
                    break
            if header_idx is not None:
                data = data[header_idx:]  # header + data rows

        if combine_two:
            data = combine_headers(data)

        out_path = os.path.join(OUTDIR, f'{out_name}.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        total_data_rows = len(data) - (1 if combine_two or sheet_name == 'PFAS Properties' else 0)
        print(f'[OK]   {out_name}.csv  →  {len(data)} rows ({total_data_rows} data + header), {len(data[0])} cols')

    zf.close()
    print('\nDone!')

if __name__ == '__main__':
    main()
