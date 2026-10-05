# Please edit the path according to your PC environment
# When setting up a new environment, run this program after executing "data_downloader.ipynb".
from pathlib import Path
import pandas as pd

YEARS = [year for year in range(2017, 2026) if year != 2020]

INPUT_BASE_DIR = Path("./data/pitch_sequence")
OUTPUT_BASE_DIR = Path("./data/pitch_type")

TARGET_COLUMNS = ["pfx_x", "pfx_z",
    "effective_speed","release_spin_rate","spin_axis"]

PITCH_TYPE_COLUMN = "pitch_type"

for year in YEARS:
    input_dir = INPUT_BASE_DIR / str(year)
    output_dir = OUTPUT_BASE_DIR / str(year)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    print(f"\n===== {year} year: {len(csv_files)} files =====")

    for csv_path in csv_files:
        player_name = csv_path.stem
        output_path = output_dir / csv_path.name
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[PITCH_TYPE_COLUMN])
        df = df[df[PITCH_TYPE_COLUMN] != "UN"]
        for col in TARGET_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate mean values for each pitch type
        result_df = (
            df
            .groupby(PITCH_TYPE_COLUMN, as_index=False)[TARGET_COLUMNS]
            .mean()
            .sort_values(PITCH_TYPE_COLUMN)
        )

        result_df.to_csv(output_path, index=False)
        print(f"[OK] {year} {player_name}: {output_path}")
