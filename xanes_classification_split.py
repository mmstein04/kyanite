"""
xanes_classification_split.py

Split a combined hand-classification CSV (columns: GrainID, Spot, Class,
where Class is 1/2/3 for pre-edge Type 1/2/3 and 4 for bad data) into one
CSV per grain, matching the pre-edge spot CSV naming used elsewhere in this
project (<grain_id>_spotNN.csv in inputs/xanes/).

Output: <grain_id>_pre_edge_classification.csv per grain, in OUTPUT_DIR —
same filename convention as the (optional, off-by-default) automatic
classifier in xanes_plot.py.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# PARAMETERS
# =============================================================================

INPUT_CSV  = '/Users/mstein/bin/kyanite/inputs/xanes_classification/xanes_classification.csv'
OUTPUT_DIR = '/Users/mstein/bin/kyanite/inputs/xanes_classification'

CATEGORY_LABELS = {1: 'Type 1', 2: 'Type 2', 3: 'Type 3', 4: 'Bad data'}

# =============================================================================

def main():
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    df = df.rename(columns={'GrainID': 'grain_id', 'Spot': 'spot', 'Class': 'category'})

    df['spot_id'] = df['grain_id'] + '_spot' + df['spot'].astype(str).str.zfill(2)
    df['category_label'] = df['category'].map(CATEGORY_LABELS)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for grain_id, g in df.groupby('grain_id'):
        g = g.sort_values('spot')[['grain_id', 'spot_id', 'spot', 'category', 'category_label']]
        out_path = out_dir / f'{grain_id}_pre_edge_classification.csv'
        g.to_csv(out_path, index=False)
        print(f'{grain_id}: {len(g)} spot(s) -> {out_path.name}  '
              f'({dict(g["category_label"].value_counts())})')


if __name__ == '__main__':
    main()
