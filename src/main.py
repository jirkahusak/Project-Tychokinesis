from pathlib import Path
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

base_dir = Path(__file__).resolve().parent.parent 
destination = base_dir / "data/processed/features/training.parquet"

pd_data = pd.read_parquet(destination, engine='pyarrow')

print(pd_data.columns)
