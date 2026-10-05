from pathlib import Path
import os
import subprocess
import sys
import time

# If this notebook is placed in the project root, no change is needed.
# Otherwise, replace Path(".") with the path to counterfactual_pitch_sequence_analysis.
project_dir = Path(".").resolve()
os.chdir(project_dir)

print(f"Working directory: {Path.cwd()}")

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "pybaseball",
    "openpyxl",
])

import pandas as pd
from pybaseball import statcast, statcast_pitcher, playerid_lookup, cache

cache.enable()

input_dir = Path("./data/stats/pitcher")
output_base_dir = Path("./data/pitch_sequence")
sample_info_path = Path("./sample_process_info.xlsx")

# Exclude the shortened 2020 season
years = [year for year in range(2017, 2026) if year != 2020]

SAMPLE_COLUMNS = [
    "process_order",
    "process_name",
    "input_sample_size",
    "output_sample_size",
    "excluded_sample_size",
    "retention_rate",
]


def split_player_name(full_name):
    parts = str(full_name).strip().split()
    first_name = parts[0]
    last_name = parts[-1]
    return first_name, last_name


def find_mlbam_id_from_name(full_name, year):
    first_name, last_name = split_player_name(full_name)
    lookup = playerid_lookup(last_name, first_name)

    if "mlb_played_first" in lookup.columns and "mlb_played_last" in lookup.columns:
        lookup_year = lookup[
            (lookup["mlb_played_first"].fillna(9999) <= year)
            & (lookup["mlb_played_last"].fillna(0) >= year)
        ]
        if not lookup_year.empty:
            lookup = lookup_year

    return int(lookup.iloc[0]["key_mlbam"])


def get_pitcher_id(row, year):
    id_columns = ["player_id", "key_mlbam", "mlbam_id", "MLBAMID"]

    for col in id_columns:
        if col in row.index and pd.notna(row[col]):
            return int(row[col])

    return find_mlbam_id_from_name(row["Name"], year)


def get_mlb_total_pitches(year):
    start_dt = f"{year}-01-01"
    end_dt = f"{year}-12-31"
    df = statcast(start_dt, end_dt)
    return len(df)


def download_pitcher_statcast(name, player_id, year, output_path):
    start_dt = f"{year}-01-01"
    end_dt = f"{year}-12-31"

    df = statcast_pitcher(start_dt, end_dt, player_id)

    if not df.empty:
        df.insert(0, "Season", year)
        df.insert(1, "Name", name)
        df.insert(2, "player_id", player_id)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(df)


def record_sample_process(
    year,
    process_order,
    process_name,
    input_sample_size,
    output_sample_size,
):
    """Record one sample-selection step in the sheet for the specified year."""

    input_sample_size = int(input_sample_size)
    output_sample_size = int(output_sample_size)

    new_row = pd.DataFrame([
        {
            "process_order": int(process_order),
            "process_name": process_name,
            "input_sample_size": input_sample_size,
            "output_sample_size": output_sample_size,
            "excluded_sample_size": input_sample_size - output_sample_size,
            "retention_rate": (
                output_sample_size / input_sample_size
                if input_sample_size > 0
                else float("nan")
            ),
        }
    ])

    sheet_name = str(year)

    if sample_info_path.exists():
        try:
            log_df = pd.read_excel(sample_info_path, sheet_name=sheet_name)
        except ValueError:
            log_df = pd.DataFrame(columns=SAMPLE_COLUMNS)
    else:
        log_df = pd.DataFrame(columns=SAMPLE_COLUMNS)

    # Re-running the same step replaces the existing row instead of duplicating it.
    if not log_df.empty:
        log_df = log_df[
            ~(
                (log_df["process_order"] == process_order)
                & (log_df["process_name"] == process_name)
            )
        ]

    log_df = pd.concat([log_df, new_row], ignore_index=True)
    log_df = log_df[SAMPLE_COLUMNS].sort_values(
        ["process_order", "process_name"]
    )

    if sample_info_path.exists():
        writer_kwargs = {
            "engine": "openpyxl",
            "mode": "a",
            "if_sheet_exists": "replace",
        }
    else:
        writer_kwargs = {
            "engine": "openpyxl",
            "mode": "w",
        }

    with pd.ExcelWriter(sample_info_path, **writer_kwargs) as writer:
        log_df.to_excel(writer, sheet_name=sheet_name, index=False)


for year in years:
    input_path = input_dir / f"{year}.xlsx"
    output_dir = output_base_dir / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitchers = pd.read_excel(input_path)

    print(f"\n===== {year} year: {len(pitchers)} pitchers =====")

    # Total number of Statcast pitches in MLB for the season.
    mlb_total_pitches = get_mlb_total_pitches(year)
    print(f"[MLB TOTAL] {year}: {mlb_total_pitches} pitches")

    # Total number of pitches thrown by the pitchers in the input file.
    target_total_pitches = 0

    for _, row in pitchers.iterrows():
        name = "_".join(str(row["Name"]).strip().split())
        filename = name + ".csv"
        output_path = output_dir / filename

        player_id = get_pitcher_id(row, year)

        n_pitches = download_pitcher_statcast(
            name=name,
            player_id=player_id,
            year=year,
            output_path=output_path,
        )

        target_total_pitches += n_pitches

        print(f"[OK] {year} {name}: {n_pitches} pitches -> {output_path}")
        time.sleep(1)

    record_sample_process(
        year=year,
        process_order=0,
        process_name="Target pitcher selection",
        input_sample_size=mlb_total_pitches,
        output_sample_size=target_total_pitches,
    )

    print(
        f"[SAMPLE] {year}: {mlb_total_pitches} -> {target_total_pitches} "
        f"({sample_info_path}, sheet={year})"
    )
