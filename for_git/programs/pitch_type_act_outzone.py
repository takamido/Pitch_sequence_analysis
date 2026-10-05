from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


YEARS = [year for year in range(2017, 2026) if year != 2020]

INPUT_BASE_DIR = Path("./data/pitch_sequence")
OUTPUT_BASE_DIR = Path("./data/pitch_type_act_outzone")

MIN_SAMPLES_PER_REGION = 5

PLATE_HALF_WIDTH_FT = 8.5 / 12.0
PITCH_TYPE_COLUMN = "pitch_type"
BALL_DESCRIPTIONS = {"ball", "blocked_ball"}

# Rank-center representative features.
# The five pitch-quality variables from pitch_type_rank_cond(2).py are retained,
# and the two model location variables are also included so that the selected
# representative pitch is central in BOTH pitch quality and location.
REPRESENTATIVE_FEATURE_COLUMNS = [
    "effective_speed", "release_spin_rate",
    "spin_axis", "pfx_x", "pfx_z",
    "plate_x", "plate_z_norm"]

# Equal weight for rank-distance score.
REPRESENTATIVE_FEATURE_WEIGHTS = {
    column: 1.0 for column in REPRESENTATIVE_FEATURE_COLUMNS}

# Eight ball regions around the strike zone, from the batter's view.
BALL_REGIONS = [
    "left", "right", "down", "up",
    "down_left", "down_right",
    "up_left", "up_right"]

print("MIN_SAMPLES_PER_REGION:", MIN_SAMPLES_PER_REGION)
print("PLATE_HALF_WIDTH_FT:", PLATE_HALF_WIDTH_FT)
print("BALL_DESCRIPTIONS:", sorted(BALL_DESCRIPTIONS))
print("BALL_REGIONS:", BALL_REGIONS)
print("REPRESENTATIVE_FEATURE_COLUMNS:", REPRESENTATIVE_FEATURE_COLUMNS)


OUTPUT_COLUMNS = [
    "pitch_type", "region", "sample_count",
    "pitch_type_total_ball_count",
    "pitch_type_min_region_count",
    "qualified_region_count", "total_region_count",
    "min_samples_per_region",
    "effective_speed", "release_spin_rate",
    "spin_axis", "pfx_x", "pfx_z",
    "plate_x", "plate_x_norm", "plate_z_norm",
    "plate_x_raw", "plate_z",
    "sz_bot", "sz_top", "stand",
    "description",
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


def add_plate_x_norm(df):
    """
    Convert batter-view plate_x in feet to the normalized horizontal coordinate
    used by analysis2_sim_multi_offset.py:
        left plate edge  -> 0.0
        right plate edge -> 1.0
        left outside     -> < 0.0
        right outside    -> > 1.0
    """
    df = df.copy()
    df["plate_x_norm"] = (
        df["plate_x"] / PLATE_HALF_WIDTH_FT + 1.0
    ) / 2.0
    return df


def assign_ball_regions(df):
    """
    Assign geometrically outside pitches to one of eight regions around the
    strike zone. Pitches whose centers are geometrically inside the rectangular
    strike zone are not assigned to a ball region, even if Statcast description
    says 'ball' or 'blocked_ball'.

    Horizontal coordinates have already been normalized to the batter's view.
    """
    df = df.copy()

    x = df["plate_x"]
    z = df["plate_z_norm"]

    inside_x = x.between(
        -PLATE_HALF_WIDTH_FT,
        PLATE_HALF_WIDTH_FT,
        inclusive="both")
    inside_z = z.between(0.0, 1.0, inclusive="both")

    left = x < -PLATE_HALF_WIDTH_FT
    right = x > PLATE_HALF_WIDTH_FT
    down = z < 0.0
    up = z > 1.0

    df["region"] = pd.NA

    df.loc[left & inside_z, "region"] = "left"
    df.loc[right & inside_z, "region"] = "right"
    df.loc[inside_x & down, "region"] = "down"
    df.loc[inside_x & up, "region"] = "up"
    df.loc[left & down, "region"] = "down_left"
    df.loc[right & down, "region"] = "down_right"
    df.loc[left & up, "region"] = "up_left"
    df.loc[right & up, "region"] = "up_right"

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

    required_columns = [
        PITCH_TYPE_COLUMN, "stand", "description",
        *numeric_columns]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[PITCH_TYPE_COLUMN])
    df = df[df[PITCH_TYPE_COLUMN] != "UN"]
    df = df[df["stand"].isin(["L", "R"])]

    # Keep exactly the ball definitions used by analysis2_sim_multi_offset.py.
    df = df[df["description"].isin(BALL_DESCRIPTIONS)].copy()

    df = add_plate_z_norm(df)
    df = normalize_plate_x_for_batter_view(df)
    df = add_plate_x_norm(df)

    required_valid_columns = [
        PITCH_TYPE_COLUMN, "stand", "description",
        *REPRESENTATIVE_FEATURE_COLUMNS,
        "plate_x", "plate_x_norm", "plate_z_norm",
        "plate_x_raw", "plate_z",
        "sz_top", "sz_bot"]
    df = df.dropna(subset=required_valid_columns).copy()

    finite_columns = [
        *REPRESENTATIVE_FEATURE_COLUMNS,
        "plate_x", "plate_x_norm", "plate_z_norm",
        "plate_x_raw", "plate_z",
        "sz_top", "sz_bot"]

    finite_mask = np.isfinite(
        df[finite_columns].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[finite_mask].copy()

    df = assign_ball_regions(df)

    # A Statcast pitch may be called a ball although its recorded center lies
    # geometrically inside the rectangular strike-zone bounds. Such pitches do
    # not belong to any of the eight outside regions and are excluded here.
    df = df.dropna(subset=["region"]).copy()
    df["region"] = pd.Categorical(
        df["region"], categories=BALL_REGIONS, ordered=True)

    return df


def find_rank_center_position(region_df):
    """
    Rank-center method copied from pitch_type_rank_cond(2).py.

    IMPORTANT: This is intentionally NOT a medoid calculation.
    For each of the seven representative features (five pitch-quality variables
    plus plate_x and plate_z_norm), rank pitches within the current
    pitcher x pitch_type x region, measure absolute distance from the center
    rank, sum those distances with equal weights, and choose the actual observed
    pitch with the minimum rank-distance score.
    """
    feature_df = region_df[REPRESENTATIVE_FEATURE_COLUMNS].astype(float)
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
    representative_position = region_df.index.get_loc(representative_index)

    return representative_position, float(rank_scores.loc[representative_index])


def get_optional_value(row, column):
    return row[column] if column in row.index else np.nan


def build_pitch_type_rank_table(raw_df):
    df = prepare_pitch_data(raw_df)

    result_rows = []
    diagnostics = []

    for pitch_type, pitch_type_df in df.groupby(
        PITCH_TYPE_COLUMN,
        sort=True,
        observed=True,
    ):
        pitch_type_df = pitch_type_df.copy()

        region_counts = (
            pitch_type_df.groupby("region", observed=True)
            .size()
            .reindex(BALL_REGIONS, fill_value=0))

        min_region_count = int(region_counts.min())
        max_region_count = int(region_counts.max())
        total_ball_count = int(region_counts.sum())
        qualified_region_mask = region_counts.ge(MIN_SAMPLES_PER_REGION)
        qualified_region_count = int(qualified_region_mask.sum())
        has_qualified_regions = qualified_region_count > 0

        diagnostics.append({
            "pitch_type": pitch_type,
            "qualified": has_qualified_regions,
            "qualified_region_count": qualified_region_count,
            "total_region_count": len(BALL_REGIONS),
            "total_ball_count": total_ball_count,
            "min_region_count": min_region_count,
            "max_region_count": max_region_count,
        })

        if not has_qualified_regions:
            continue

        for region in BALL_REGIONS:
            region_count = int(region_counts.loc[region])
            if region_count < MIN_SAMPLES_PER_REGION:
                continue

            region_df = pitch_type_df[
                pitch_type_df["region"] == region
            ].copy()

            # Rank-center selection: unchanged in principle from the original.
            representative_position, rank_center_score = (
                find_rank_center_position(region_df))
            representative = region_df.iloc[representative_position]

            result_rows.append({
                "pitch_type": pitch_type,
                "region": region,
                "sample_count": region_count,
                "pitch_type_total_ball_count": total_ball_count,
                "pitch_type_min_region_count": min_region_count,
                "qualified_region_count": qualified_region_count,
                "total_region_count": len(BALL_REGIONS),
                "min_samples_per_region": MIN_SAMPLES_PER_REGION,
                "effective_speed": representative["effective_speed"],
                "release_spin_rate": representative["release_spin_rate"],
                "spin_axis": representative["spin_axis"],
                "pfx_x": representative["pfx_x"],
                "pfx_z": representative["pfx_z"],
                "plate_x": representative["plate_x"],
                "plate_x_norm": representative["plate_x_norm"],
                "plate_z_norm": representative["plate_z_norm"],
                "plate_x_raw": representative["plate_x_raw"],
                "plate_z": representative["plate_z"],
                "sz_bot": representative["sz_bot"],
                "sz_top": representative["sz_top"],
                "stand": representative["stand"],
                "description": representative["description"],
                "source_row_index": representative["source_row_index"],
                "game_date": get_optional_value(representative, "game_date"),
                "game_pk": get_optional_value(representative, "game_pk"),
                "at_bat_number": get_optional_value(
                    representative, "at_bat_number"),
                "pitch_number": get_optional_value(representative, "pitch_number"),
                "batter": get_optional_value(representative, "batter"),
                "rank_center_score": rank_center_score,
            })

    result_df = pd.DataFrame(result_rows, columns=OUTPUT_COLUMNS)
    if not result_df.empty:
        region_order = {region: i for i, region in enumerate(BALL_REGIONS)}
        result_df["_region_order"] = result_df["region"].map(region_order)
        result_df = (
            result_df.sort_values(["pitch_type", "_region_order"])
            .drop(columns=["_region_order"])
            .reset_index(drop=True))

    diagnostics_df = pd.DataFrame(diagnostics)
    return result_df, diagnostics_df


META_PITCHER_COLUMNS = [
    "year", "player_name",
    "input_file", "status",
    "error_message", "raw_pitch_count",
    "valid_ball_pitch_count",
    "candidate_pitch_type_count",
    "extracted_pitch_type_count",
    "excluded_pitch_type_count",
    "pitch_type_extraction_rate",
    "candidate_pitch_type_region_count",
    "extracted_pitch_type_region_count",
    "region_extraction_rate",
    "output_row_count",
    "candidate_pitch_types",
    "extracted_pitch_types",
    "excluded_pitch_types",
    "total_region_count", "min_samples_per_region",
    "output_csv_path"]


META_PITCH_TYPE_COLUMNS = [
    "year", "player_name",
    "input_file", "pitch_type",
    "extracted", "qualified_region_count",
    "total_region_count", "region_extraction_rate",
    "total_ball_count", "min_region_count",
    "max_region_count",
    "min_samples_per_region", "output_csv_path"]


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
        valid_ball_pitch_count = 0
        extracted_region_count = 0
    else:
        candidate_pitch_types = diagnostics_df["pitch_type"].tolist()
        extracted_pitch_types = diagnostics_df.loc[
            diagnostics_df["qualified"], "pitch_type"].tolist()
        excluded_pitch_types = diagnostics_df.loc[
            ~diagnostics_df["qualified"], "pitch_type"].tolist()
        valid_ball_pitch_count = int(diagnostics_df["total_ball_count"].sum())
        extracted_region_count = int(
            diagnostics_df["qualified_region_count"].sum())

    candidate_pitch_type_count = len(candidate_pitch_types)
    extracted_pitch_type_count = len(extracted_pitch_types)
    candidate_region_count = candidate_pitch_type_count * len(BALL_REGIONS)

    pitch_type_extraction_rate = (
        extracted_pitch_type_count / candidate_pitch_type_count
        if candidate_pitch_type_count > 0
        else np.nan)

    region_extraction_rate = (
        extracted_region_count / candidate_region_count
        if candidate_region_count > 0
        else np.nan)

    return {
        "year": year,
        "player_name": player_name,
        "input_file": str(input_file),
        "status": "ok",
        "error_message": "",
        "raw_pitch_count": int(raw_pitch_count),
        "valid_ball_pitch_count": valid_ball_pitch_count,
        "candidate_pitch_type_count": candidate_pitch_type_count,
        "extracted_pitch_type_count": extracted_pitch_type_count,
        "excluded_pitch_type_count": (
            candidate_pitch_type_count - extracted_pitch_type_count),
        "pitch_type_extraction_rate": pitch_type_extraction_rate,
        "candidate_pitch_type_region_count": candidate_region_count,
        "extracted_pitch_type_region_count": extracted_region_count,
        "region_extraction_rate": region_extraction_rate,
        "output_row_count": int(output_row_count),
        "candidate_pitch_types": _join_pitch_types(candidate_pitch_types),
        "extracted_pitch_types": _join_pitch_types(extracted_pitch_types),
        "excluded_pitch_types": _join_pitch_types(excluded_pitch_types),
        "total_region_count": len(BALL_REGIONS),
        "min_samples_per_region": MIN_SAMPLES_PER_REGION,
        "output_csv_path": str(output_csv_path),
    }


def build_pitch_type_meta_rows(
    year, player_name,
    input_file, output_csv_path, diagnostics_df,
):
    rows = []
    for row in diagnostics_df.itertuples(index=False):
        rows.append({
            "year": year,
            "player_name": player_name,
            "input_file": str(input_file),
            "pitch_type": row.pitch_type,
            "extracted": bool(row.qualified),
            "qualified_region_count": int(row.qualified_region_count),
            "total_region_count": int(row.total_region_count),
            "region_extraction_rate": (
                row.qualified_region_count / row.total_region_count
                if row.total_region_count > 0
                else np.nan),
            "total_ball_count": int(row.total_ball_count),
            "min_region_count": int(row.min_region_count),
            "max_region_count": int(row.max_region_count),
            "min_samples_per_region": MIN_SAMPLES_PER_REGION,
            "output_csv_path": str(output_csv_path),
        })

    return rows


def write_year_metadata(year, pitcher_rows, pitch_type_rows):
    meta_path = OUTPUT_BASE_DIR / f"meta_{year}.xlsx"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    pitcher_meta_df = pd.DataFrame(
        pitcher_rows,
        columns=META_PITCHER_COLUMNS)
    pitch_type_meta_df = pd.DataFrame(
        pitch_type_rows,
        columns=META_PITCH_TYPE_COLUMNS)

    if not pitcher_meta_df.empty:
        pitcher_meta_df = pitcher_meta_df.sort_values(
            ["status", "player_name"],
            ascending=[True, True]).reset_index(drop=True)

    if not pitch_type_meta_df.empty:
        pitch_type_meta_df = pitch_type_meta_df.sort_values(
            ["player_name", "pitch_type"]).reset_index(drop=True)

    with pd.ExcelWriter(meta_path) as writer:
        pitcher_meta_df.to_excel(
            writer,
            sheet_name="pitcher_summary",
            index=False,
            freeze_panes=(1, 0))

        pitch_type_meta_df.to_excel(
            writer,
            sheet_name="pitch_type_detail",
            index=False,
            freeze_panes=(1, 0))

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
            int(diagnostics_df["qualified_region_count"].sum())
            if not diagnostics_df.empty
            else 0)

        if len(result_df) != expected_output_rows:
            raise RuntimeError(
                f"Output row mismatch for {year} {player_name}: "
                f"expected={expected_output_rows}, actual={len(result_df)}")

        result_df.to_csv(output_path, index=False)

        year_pitcher_meta_rows.append(
            build_pitcher_meta_row(
                year=year,
                player_name=player_name,
                input_file=csv_path,
                output_csv_path=output_path,
                raw_pitch_count=raw_pitch_count,
                diagnostics_df=diagnostics_df,
                output_row_count=len(result_df)))

        year_pitch_type_meta_rows.extend(
            build_pitch_type_meta_rows(
                year=year,
                player_name=player_name,
                input_file=csv_path,
                output_csv_path=output_path,
                diagnostics_df=diagnostics_df))

        qualified_pitch_type_count = (
            int(diagnostics_df["qualified"].sum())
            if not diagnostics_df.empty
            else 0)

        qualified_region_count = expected_output_rows
        examined_pitch_type_count = len(diagnostics_df)
        examined_region_count = examined_pitch_type_count * len(BALL_REGIONS)

        print(
            f"[OK] {year} {player_name}: "
            f"pitch types with output={qualified_pitch_type_count}/"
            f"{examined_pitch_type_count}, "
            f"qualified regions={qualified_region_count}/{examined_region_count}, "
            f"output rows={len(result_df)} -> {output_path}")

        if not diagnostics_df.empty:
            excluded = diagnostics_df.loc[
                diagnostics_df["qualified_region_count"].eq(0)]

            for row in excluded.itertuples(index=False):
                print(
                    f"  [SKIP] {row.pitch_type}: "
                    f"no ball region reached {MIN_SAMPLES_PER_REGION} pitches, "
                    f"max region count={row.max_region_count}, "
                    f"total ball pitches={row.total_ball_count}")

            partial = diagnostics_df.loc[
                diagnostics_df["qualified_region_count"].between(
                    1, len(BALL_REGIONS) - 1)]

            for row in partial.itertuples(index=False):
                print(
                    f"  [PARTIAL] {row.pitch_type}: "
                    f"output regions={row.qualified_region_count}/"
                    f"{row.total_region_count}, "
                    f"required per region={MIN_SAMPLES_PER_REGION}, "
                    f"total ball pitches={row.total_ball_count}")

    meta_path = write_year_metadata(
        year=year,
        pitcher_rows=year_pitcher_meta_rows,
        pitch_type_rows=year_pitch_type_meta_rows)

    ok_count = sum(row["status"] == "ok" for row in year_pitcher_meta_rows)
    print(
        f"[META] {year}: pitchers={len(year_pitcher_meta_rows)}, "
        f"ok={ok_count}, meta={meta_path}")

print("Finish all conditional ball-region rank-center extraction")
