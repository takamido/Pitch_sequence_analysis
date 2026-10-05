from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

YEARS = [year for year in range(2017, 2026) if year != 2020]

INPUT_BASE_DIR = Path("./data/pitch_sequence")
OUTPUT_BASE_DIR = Path("./data/pitch_type_act_inzone")

N_GRID = 3
MIN_SAMPLES_PER_CELL = 5

PLATE_HALF_WIDTH_FT = 8.5 / 12.0
PITCH_TYPE_COLUMN = "pitch_type"

REPRESENTATIVE_FEATURE_COLUMNS = [
    "effective_speed", "release_spin_rate",
    "spin_axis", "pfx_x", "pfx_z",
    "plate_x", "plate_z_norm"]

# Equal weight for rank-distance score.
REPRESENTATIVE_FEATURE_WEIGHTS = {
    column: 1.0 for column in REPRESENTATIVE_FEATURE_COLUMNS}

# Grid edges and centers. These define N_GRID cells, not N_GRID evaluation points.
GRID_X_EDGES = np.linspace(
    -PLATE_HALF_WIDTH_FT, PLATE_HALF_WIDTH_FT,
    N_GRID + 1)

GRID_Z_EDGES = np.linspace(0.0, 1.0, N_GRID + 1)

GRID_X_CENTERS = (GRID_X_EDGES[:-1] + GRID_X_EDGES[1:]) / 2.0
GRID_Z_CENTERS = (GRID_Z_EDGES[:-1] + GRID_Z_EDGES[1:]) / 2.0
GRID_X_CENTERS[np.isclose(GRID_X_CENTERS, 0.0)] = 0.0
GRID_Z_CENTERS[np.isclose(GRID_Z_CENTERS, 0.0)] = 0.0

print("N_GRID:", N_GRID)
print("MIN_SAMPLES_PER_CELL:", MIN_SAMPLES_PER_CELL)
print("GRID_X_EDGES:", GRID_X_EDGES)
print("GRID_X_CENTERS:", GRID_X_CENTERS)
print("GRID_Z_EDGES:", GRID_Z_EDGES)
print("GRID_Z_CENTERS:", GRID_Z_CENTERS)

OUTPUT_COLUMNS = [
    "pitch_type", "grid_x_index",
    "grid_z_index", "grid_x_left",
    "grid_x_right", "grid_z_lower",
    "grid_z_upper", "grid_x_center",
    "grid_z_center", "sample_count",
    "pitch_type_total_in_grid",
    "pitch_type_min_cell_count",
    "n_grid", "min_samples_per_cell",
    "effective_speed", "release_spin_rate",
    "spin_axis", "pfx_x", "pfx_z",
    "plate_x", "plate_z_norm",
    "plate_x_raw", "plate_z",
    "sz_bot", "sz_top", "stand", 
    "source_row_index", "game_date",
    "game_pk", "at_bat_number",
    "pitch_number", "batter", "rank_center_score"]

def add_plate_z_norm(df):
    df = df.copy()
    zone_height = df["sz_top"] - df["sz_bot"]
    valid_height = zone_height > 0
    df["plate_z_norm"] = np.nan
    df.loc[valid_height, "plate_z_norm"] = (
        (df.loc[valid_height, "plate_z"] - df.loc[valid_height, "sz_bot"])
        / zone_height.loc[valid_height])
    return df

def normalize_plate_x_for_batter_view(df):
    df = df.copy()
    df["plate_x_raw"] = df["plate_x"]
    left_mask = df["stand"].eq("L")
    df.loc[left_mask, "plate_x"] = -df.loc[left_mask, "plate_x"]
    return df

def assign_grid_cells(df):
    df = df.copy()
    inside = (
        df["plate_x"].between(GRID_X_EDGES[0], GRID_X_EDGES[-1], inclusive="both")
        & df["plate_z_norm"].between(
            GRID_Z_EDGES[0], GRID_Z_EDGES[-1], inclusive="both"))

    df = df.loc[inside].copy()
    df["grid_x_index"] = (np.searchsorted(GRID_X_EDGES, df["plate_x"].to_numpy(), side="right") - 1)
    df["grid_z_index"] = (np.searchsorted(GRID_Z_EDGES, df["plate_z_norm"].to_numpy(), side="right") - 1)
    df["grid_x_index"] = df["grid_x_index"].clip(0, N_GRID - 1).astype(int)
    df["grid_z_index"] = df["grid_z_index"].clip(0, N_GRID - 1).astype(int)
    return df


def prepare_pitch_data(raw_df):
    df = raw_df.copy()
    df["source_row_index"] = np.arange(len(df), dtype=int)

    numeric_columns = [
        "plate_x", "plate_z",
        "sz_top", "sz_bot",
        "effective_speed",
        "release_spin_rate",
        "spin_axis", "pfx_x", "pfx_z"]

    optional_numeric_columns = [
        "game_pk", "at_bat_number",
        "pitch_number", "batter"]

    for column in numeric_columns + optional_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[PITCH_TYPE_COLUMN])
    df = df[df[PITCH_TYPE_COLUMN] != "UN"]
    df = df[df["stand"].isin(["L", "R"])]

    df = add_plate_z_norm(df)
    df = normalize_plate_x_for_batter_view(df)

    required_valid_columns = [
        PITCH_TYPE_COLUMN, "stand",
        *REPRESENTATIVE_FEATURE_COLUMNS,
        "plate_x_raw", "plate_z",
        "sz_top", "sz_bot"]
    df = df.dropna(subset=required_valid_columns).copy()

    finite_columns = [
        *REPRESENTATIVE_FEATURE_COLUMNS,
        "plate_x_raw", "plate_z",
        "sz_top", "sz_bot"]

    finite_mask = np.isfinite(df[finite_columns].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[finite_mask].copy()

    return assign_grid_cells(df)

def find_rank_center_position(cell_df):
    feature_df = cell_df[REPRESENTATIVE_FEATURE_COLUMNS].astype(float)
    n_samples = len(feature_df)
    center_rank = (n_samples + 1) / 2.0

    ranks = feature_df.rank(method="average", axis=0)
    rank_distances = (ranks - center_rank).abs()

    weights = pd.Series(
        REPRESENTATIVE_FEATURE_WEIGHTS,
        index=REPRESENTATIVE_FEATURE_COLUMNS,
        dtype=float)

    rank_scores = rank_distances.mul(weights, axis=1).sum(axis=1)
    representative_index = rank_scores.idxmin()
    representative_position = cell_df.index.get_loc(representative_index)

    return representative_position, float(rank_scores.loc[representative_index])

def get_optional_value(row, column):
    return row[column] if column in row.index else np.nan

def build_pitch_type_rank_table(raw_df):
    df = prepare_pitch_data(raw_df)
    expected_cells = pd.MultiIndex.from_product(
        [range(N_GRID), range(N_GRID)],
        names=["grid_x_index", "grid_z_index"])

    result_rows = []; diagnostics = []

    for pitch_type, pitch_type_df in df.groupby(PITCH_TYPE_COLUMN, sort=True):
        pitch_type_df = pitch_type_df.copy()

        cell_counts = (pitch_type_df.groupby(["grid_x_index", "grid_z_index"]).size().reindex(expected_cells, fill_value=0))

        min_cell_count = int(cell_counts.min())
        max_cell_count = int(cell_counts.max())
        total_in_grid = int(cell_counts.sum())
        qualified_cell_mask = cell_counts.ge(MIN_SAMPLES_PER_CELL)
        qualified_cell_count = int(qualified_cell_mask.sum())
        has_qualified_cells = qualified_cell_count > 0

        diagnostics.append({
            "pitch_type": pitch_type,
            "qualified": has_qualified_cells,
            "qualified_cell_count": qualified_cell_count,
            "total_cell_count": N_GRID ** 2,
            "total_in_grid": total_in_grid,
            "min_cell_count": min_cell_count,
            "max_cell_count": max_cell_count,
        })

        if not has_qualified_cells:
           continue

        for grid_x_index in range(N_GRID):
            for grid_z_index in range(N_GRID):
                cell_count = int(cell_counts.loc[(grid_x_index, grid_z_index)])
                if cell_count < MIN_SAMPLES_PER_CELL:
                   continue

                cell_df = pitch_type_df[
                    (pitch_type_df["grid_x_index"] == grid_x_index)
                    & (pitch_type_df["grid_z_index"] == grid_z_index)].copy()

                representative_position, rank_center_score = (
                    find_rank_center_position(cell_df))
                representative = cell_df.iloc[representative_position]

                result_rows.append({
                    "pitch_type": pitch_type,
                    "grid_x_index": grid_x_index,
                    "grid_z_index": grid_z_index,
                    "grid_x_left": GRID_X_EDGES[grid_x_index],
                    "grid_x_right": GRID_X_EDGES[grid_x_index + 1],
                    "grid_z_lower": GRID_Z_EDGES[grid_z_index],
                    "grid_z_upper": GRID_Z_EDGES[grid_z_index + 1],
                    "grid_x_center": GRID_X_CENTERS[grid_x_index],
                    "grid_z_center": GRID_Z_CENTERS[grid_z_index],
                    "sample_count": cell_count,
                    "pitch_type_total_in_grid": total_in_grid,
                    "pitch_type_min_cell_count": min_cell_count,
                    "n_grid": N_GRID,
                    "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
                    "effective_speed": representative["effective_speed"],
                    "release_spin_rate": representative["release_spin_rate"],
                    "spin_axis": representative["spin_axis"],
                    "pfx_x": representative["pfx_x"],
                    "pfx_z": representative["pfx_z"],
                    "plate_x": representative["plate_x"],
                    "plate_z_norm": representative["plate_z_norm"],
                    "plate_x_raw": representative["plate_x_raw"],
                    "plate_z": representative["plate_z"],
                    "sz_bot": representative["sz_bot"],
                    "sz_top": representative["sz_top"],
                    "stand": representative["stand"],
                    "source_row_index": representative["source_row_index"],
                    "game_date": get_optional_value(representative, "game_date"),
                    "game_pk": get_optional_value(representative, "game_pk"),
                    "at_bat_number": get_optional_value(representative, "at_bat_number"),
                    "pitch_number": get_optional_value(representative, "pitch_number"),
                    "batter": get_optional_value(representative, "batter"),
                    "rank_center_score": rank_center_score})

    result_df = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)
    if not result_df.empty:
        result_df = result_df.sort_values(["pitch_type", "grid_x_index", "grid_z_index"]).reset_index(drop=True)

    diagnostics_df = pd.DataFrame(diagnostics)
    return result_df, diagnostics_df

META_PITCHER_COLUMNS = [
    "year", "player_name",
    "input_file", "status",
    "error_message", "raw_pitch_count",
    "valid_in_grid_pitch_count",
    "candidate_pitch_type_count",
    "extracted_pitch_type_count",
    "excluded_pitch_type_count",
    "pitch_type_extraction_rate",
    "candidate_pitch_type_cell_count",
    "extracted_pitch_type_cell_count",
    "cell_extraction_rate",
    "output_row_count",
    "candidate_pitch_types",
    "extracted_pitch_types",
    "excluded_pitch_types",
    "n_grid", "min_samples_per_cell",
    "output_csv_path"]

META_PITCH_TYPE_COLUMNS = [
    "year", "player_name",
    "input_file", "pitch_type",
    "extracted", "qualified_cell_count",
    "total_cell_count", "cell_extraction_rate",
    "total_in_grid", "min_cell_count",
    "max_cell_count", "n_grid",
    "min_samples_per_cell", "output_csv_path"]

def _join_pitch_types(values):
    return ", ".join(sorted(str(value) for value in values))

def build_pitcher_meta_row(
    year, player_name,
    input_file, output_csv_path,
    raw_pitch_count, diagnostics_df,
    output_row_count,
):
    if diagnostics_df.empty:
        candidate_pitch_types = []
        extracted_pitch_types = []
        excluded_pitch_types = []
        valid_in_grid_pitch_count = 0
        extracted_cell_count = 0
    else:
        candidate_pitch_types = diagnostics_df["pitch_type"].tolist()
        extracted_pitch_types = diagnostics_df.loc[diagnostics_df["qualified"], "pitch_type"].tolist()
        excluded_pitch_types = diagnostics_df.loc[~diagnostics_df["qualified"], "pitch_type"].tolist()
        valid_in_grid_pitch_count = int(diagnostics_df["total_in_grid"].sum())
        extracted_cell_count = int(diagnostics_df["qualified_cell_count"].sum())

    candidate_pitch_type_count = len(candidate_pitch_types)
    extracted_pitch_type_count = len(extracted_pitch_types)
    candidate_cell_count = candidate_pitch_type_count * (N_GRID ** 2)

    pitch_type_extraction_rate = (
        extracted_pitch_type_count / candidate_pitch_type_count
        if candidate_pitch_type_count > 0
        else np.nan)

    cell_extraction_rate = (
        extracted_cell_count / candidate_cell_count
        if candidate_cell_count > 0
        else np.nan)

    return {
        "year": year, "player_name": player_name,
        "input_file": str(input_file),
        "status": "ok", "error_message": "",
        "raw_pitch_count": int(raw_pitch_count),
        "valid_in_grid_pitch_count": valid_in_grid_pitch_count,
        "candidate_pitch_type_count": candidate_pitch_type_count,
        "extracted_pitch_type_count": extracted_pitch_type_count,
        "excluded_pitch_type_count": (candidate_pitch_type_count - extracted_pitch_type_count),
        "pitch_type_extraction_rate": pitch_type_extraction_rate,
        "candidate_pitch_type_cell_count": candidate_cell_count,
        "extracted_pitch_type_cell_count": extracted_cell_count,
        "cell_extraction_rate": cell_extraction_rate,
        "output_row_count": int(output_row_count),
        "candidate_pitch_types": _join_pitch_types(candidate_pitch_types),
        "extracted_pitch_types": _join_pitch_types(extracted_pitch_types),
        "excluded_pitch_types": _join_pitch_types(excluded_pitch_types),
        "n_grid": N_GRID, "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
        "output_csv_path": str(output_csv_path)}

def build_pitch_type_meta_rows(
    year, player_name,
    input_file, output_csv_path, diagnostics_df,
):
    rows = []
    for row in diagnostics_df.itertuples(index=False):
        rows.append({
            "year": year, "player_name": player_name,
            "input_file": str(input_file),
            "pitch_type": row.pitch_type,
            "extracted": bool(row.qualified),
            "qualified_cell_count": int(row.qualified_cell_count),
            "total_cell_count": int(row.total_cell_count),
            "cell_extraction_rate": (
                row.qualified_cell_count / row.total_cell_count
                if row.total_cell_count > 0
                else np.nan),
            "total_in_grid": int(row.total_in_grid),
            "min_cell_count": int(row.min_cell_count),
            "max_cell_count": int(row.max_cell_count),
            "n_grid": N_GRID,
            "min_samples_per_cell": MIN_SAMPLES_PER_CELL,
            "output_csv_path": str(output_csv_path)})
    
    return rows

def write_year_metadata(year, pitcher_rows, pitch_type_rows):
    meta_path = OUTPUT_BASE_DIR / f"meta_{year}.xlsx"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pitcher_meta_df = pd.DataFrame(pitcher_rows, columns=META_PITCHER_COLUMNS)
    pitch_type_meta_df = pd.DataFrame(
        pitch_type_rows,
        columns=META_PITCH_TYPE_COLUMNS)

    if not pitcher_meta_df.empty:
        pitcher_meta_df = pitcher_meta_df.sort_values(
            ["status", "player_name"], ascending=[True, True]).reset_index(drop=True)

    if not pitch_type_meta_df.empty:
        pitch_type_meta_df = pitch_type_meta_df.sort_values(
            ["player_name", "pitch_type"]).reset_index(drop=True)

    with pd.ExcelWriter(meta_path) as writer:
        pitcher_meta_df.to_excel(
            writer, sheet_name="pitcher_summary",
            index=False, freeze_panes=(1, 0))

        pitch_type_meta_df.to_excel(
            writer, sheet_name="pitch_type_detail",
            index=False, freeze_panes=(1, 0))

    return meta_path

for year in YEARS:
    input_dir = INPUT_BASE_DIR / str(year)
    output_dir = OUTPUT_BASE_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"\n===== {year}: {len(csv_files)} files =====")

    year_pitcher_meta_rows = []
    year_pitch_type_meta_rows = []

    for csv_path in tqdm(csv_files, desc=str(year)):
        player_name = csv_path.stem
        output_path = output_dir / csv_path.name
        raw_pitch_count = np.nan

        raw_df = pd.read_csv(csv_path)
        raw_pitch_count = len(raw_df)
        result_df, diagnostics_df = build_pitch_type_rank_table(raw_df)

        expected_output_rows = (
            int(diagnostics_df["qualified_cell_count"].sum())
            if not diagnostics_df.empty
            else 0)

        result_df.to_csv(output_path, index=False)

        year_pitcher_meta_rows.append(
            build_pitcher_meta_row(
                year=year, player_name=player_name,
                input_file=csv_path, output_csv_path=output_path,
                raw_pitch_count=raw_pitch_count, diagnostics_df=diagnostics_df,
                output_row_count=len(result_df)))

        year_pitch_type_meta_rows.extend(
            build_pitch_type_meta_rows(
                year=year, player_name=player_name,
                input_file=csv_path, output_csv_path=output_path,
                diagnostics_df=diagnostics_df))

        qualified_pitch_type_count = (
            int(diagnostics_df["qualified"].sum())
            if not diagnostics_df.empty
            else 0)

        qualified_cell_count = expected_output_rows
        examined_pitch_type_count = len(diagnostics_df)
        examined_cell_count = examined_pitch_type_count * (N_GRID ** 2)

        print(
            f"[OK] {year} {player_name}: "
            f"pitch types with output={qualified_pitch_type_count}/"
            f"{examined_pitch_type_count}, "
            f"qualified cells={qualified_cell_count}/{examined_cell_count}, "
            f"output rows={len(result_df)} -> {output_path}")
            

        if not diagnostics_df.empty:
            excluded = diagnostics_df.loc[
                diagnostics_df["qualified_cell_count"].eq(0)]

            for row in excluded.itertuples(index=False):
                print(
                    f"  [SKIP] {row.pitch_type}: "
                    f"no cell reached {MIN_SAMPLES_PER_CELL} pitches, "
                    f"max cell count={row.max_cell_count}, "
                    f"total in grid={row.total_in_grid}")

            partial = diagnostics_df.loc[
                diagnostics_df["qualified_cell_count"].between(1, N_GRID ** 2 - 1)]

            for row in partial.itertuples(index=False):
                print(
                    f"  [PARTIAL] {row.pitch_type}: "
                    f"output cells={row.qualified_cell_count}/{row.total_cell_count}, "
                    f"required per cell={MIN_SAMPLES_PER_CELL}, "
                    f"total in grid={row.total_in_grid}")

    meta_path = write_year_metadata(
        year=year,
        pitcher_rows=year_pitcher_meta_rows,
        pitch_type_rows=year_pitch_type_meta_rows)
    ok_count = sum(row["status"] == "ok" for row in year_pitcher_meta_rows)
    print(f"[META] {year}: pitchers={len(year_pitcher_meta_rows)}, ")

print("Finish all conditional rank-center extraction")
