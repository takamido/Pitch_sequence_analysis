from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

DATA_ROOT = Path("./data/pitch_sequence")
BATTER_STATS_DATA_ROOT = Path("./data/stats/batter")
SAMPLE_INFO_PATH = Path("./sample_process_info.xlsx")

SAMPLE_COLUMNS = [
    "process_order",
    "process_name",
    "input_sample_size",
    "output_sample_size",
    "excluded_sample_size",
    "retention_rate",
]

# Set the target years to 2018 and later because previous-year batter statistics are required.
YEAR_SPLITS = [
    ([2018], [2019]),
    ([2018, 2019], [2022]),
    ([2018, 2019, 2022], [2023]),
    ([2018, 2019, 2022, 2023], [2024]),
    ([2018, 2019, 2022, 2023, 2024], [2025])]

# Analyze up to five pitches before the final pitch
MAX_PITCHES = 6
TARGET_COLUMN = "estimated_slg_using_speedangle"

BALL_DESCRIPTIONS = {"ball", "blocked_ball"}

# 1 = immediately previous pitch, 2 = two pitches before the last pitch, ...
BALL_OFFSETS = [1, 2, 3, 4, 5]

# Feature variables
PITCH_FEATURE_COLUMNS = ["effective_speed", "release_spin_rate","spin_axis", "pfx_x",
    "pfx_z", "plate_x","plate_z_norm"]

# Context information
CONTEXT_STATIC_COLUMNS = ["balls", "outs_when_up", "inning", "is_top_inning",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "bat_score_diff", "n_thruorder_pitcher"]

BATTER_STAT_COLUMNS = ["K_rate", "HR_rate", "ISO", "AVG", "OPS"]
BAD_FALLBACK_MULTIPLIER = {"K_rate": 1.10,"HR_rate": 0.90,"ISO": 0.90,"AVG": 0.90,"OPS": 0.90}

C_FEATURE_COLUMNS = (BATTER_STAT_COLUMNS
    + ["used_batter_stats_fallback"]
    + CONTEXT_STATIC_COLUMNS)

# Dataset definitions.
# for sensitivity analysis, also extract plate appearances in which the pitch 1~5 pitches earlier was a ball.
VARIANT_CONFIGS = {
    "normal": {"label": "last-pitch","ball_offset": None,
        "output_dir": Path("./dataset/last"),
        "file_stem": "last"
        },
    "ball_1_before": {"label": "1 pitch before last = ball/blocked_ball",
        "ball_offset": 1,"output_dir": Path("./dataset/preceding"),
        "file_stem": "preceding_1",
        },
    "ball_2_before": {"label": "2 pitches before last = ball/blocked_ball",
        "ball_offset": 2,"output_dir": Path("./dataset/preceding"),
        "file_stem": "preceding_2",
        },
    "ball_3_before": {"label": "3 pitches before last = ball/blocked_ball",
        "ball_offset": 3,"output_dir": Path("./dataset/preceding"),
        "file_stem": "preceding_3",
        },
    "ball_4_before": {"label": "4 pitches before last = ball/blocked_ball",
        "ball_offset": 4,"output_dir": Path("./dataset/preceding"),
        "file_stem": "preceding_4",
        },
    "ball_5_before": {"label": "5 pitches before last = ball/blocked_ball",
        "ball_offset": 5,"output_dir": Path("./dataset/preceding"),
        "file_stem": "preceding_5",
        }}

# Screening stages. Pitch counts after the raw stage are the total number of
# pitch rows belonging to plate appearances that survive each stage.
SCREENING_STAGE_DEFS = [
    ("raw", "Raw input", None),
    ("last_pitch_two_strikes", "Last pitch starts with 2 strikes", "raw"),
    ("valid_target", "Target is swinging strike or valid hit_into_play xSLG", "last_pitch_two_strikes"),
    ("complete_pitch_features", "No missing pitch features in the plate appearance", "valid_target"),
    ("complete_context", "No missing context features on the last pitch", "complete_pitch_features"),
    ("final_normal", "Accepted into normal dataset", "complete_context"),
] + [
    (
        f"ball_{offset}_before",
        f"Accepted normal sample and {offset} pitch(es) before last is ball/blocked_ball",
        "final_normal",
    )
    for offset in BALL_OFFSETS
]

# Definition of the functions
def add_plate_z_norm(df):
    df = df.copy()
    df["plate_z_norm"] = (
        (df["plate_z"] - df["sz_bot"])
        / (df["sz_top"] - df["sz_bot"]))
    return df

def normalize_x_for_batter_view(df):
    df = df.copy()
    left_mask = df["stand"] == "L"
    for col in ["pfx_x", "plate_x"]:
        if col in df.columns:
           df.loc[left_mask, col] = -df.loc[left_mask, col]
    return df

def add_context_features(df):
    df = df.copy()
    df["runner_on_1b"] = df["on_1b"].notna().astype(float)
    df["runner_on_2b"] = df["on_2b"].notna().astype(float)
    df["runner_on_3b"] = df["on_3b"].notna().astype(float)
    df["is_top_inning"] = (df["inning_topbot"] == "Top").astype(float)
    return df

def preprocess_pitch_df(df):
    df = df.copy()
    df = add_plate_z_norm(df)
    df = normalize_x_for_batter_view(df)
    df = add_context_features(df)
    return df

def pad_sequence(seq, max_len=6):
    if len(seq) >= max_len:
        return seq[-max_len:]
    padded = np.zeros((max_len, seq.shape[1]), dtype=np.float32)
    padded[-len(seq):] = seq
    return padded

def get_target(last_pitch):
    desc = last_pitch["description"]
    if desc in ["swinging_strike", "swinging_strike_blocked"]:
        return 0.0
    if desc == "hit_into_play":
        xslg = last_pitch[TARGET_COLUMN]
        if pd.isna(xslg) or not np.isfinite(xslg):
            return None
        return float(xslg)
    return None

def pick_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def make_worse_fallback_stats(base_stats):
    base_stats = np.asarray(base_stats, dtype=np.float32).copy()
    adjusted = base_stats.copy()
    for i, col in enumerate(BATTER_STAT_COLUMNS):
        adjusted[i] = adjusted[i] * BAD_FALLBACK_MULTIPLIER[col]

    adjusted[BATTER_STAT_COLUMNS.index("K_rate")] = np.clip(
        adjusted[BATTER_STAT_COLUMNS.index("K_rate")], 0.0, 1.0)
    adjusted[BATTER_STAT_COLUMNS.index("HR_rate")] = np.clip(
        adjusted[BATTER_STAT_COLUMNS.index("HR_rate")], 0.0, 1.0)
    adjusted[BATTER_STAT_COLUMNS.index("ISO")] = max(
        adjusted[BATTER_STAT_COLUMNS.index("ISO")], 0.0)
    adjusted[BATTER_STAT_COLUMNS.index("AVG")] = np.clip(
        adjusted[BATTER_STAT_COLUMNS.index("AVG")], 0.0, 1.0)
    adjusted[BATTER_STAT_COLUMNS.index("OPS")] = max(
        adjusted[BATTER_STAT_COLUMNS.index("OPS")], 0.0)

    return adjusted.astype(np.float32)

def load_batter_stat_info(prev_year):
    excel_path = BATTER_STATS_DATA_ROOT / f"{prev_year}.xlsx"
    df = pd.read_excel(excel_path)
    df = df.copy()

    print(f"\nloading batter stats: {excel_path}")
    print("columns:", df.columns.tolist())

    id_col = pick_first_existing_column(df, candidates=["player_id"])
    name_col = pick_first_existing_column(df, candidates=["Name"])
    pa_col = pick_first_existing_column(df, candidates=["plateAppearances"])
    so_col = pick_first_existing_column(df, candidates=["strikeOuts"])
    hr_col = pick_first_existing_column(df, candidates=["homeRuns"])
    avg_col = pick_first_existing_column(df, candidates=["avg"])
    slg_col = pick_first_existing_column(df, candidates=["slg"])
    ops_col = pick_first_existing_column(df, candidates=["ops"])

    numeric_cols = [id_col, pa_col, so_col, hr_col, avg_col, slg_col, ops_col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    player_df = df.copy()
    player_df = player_df.dropna(subset=[id_col, pa_col, so_col, hr_col, avg_col, slg_col, ops_col])
    player_df[id_col] = player_df[id_col].astype(int)
    player_df = player_df[player_df[pa_col] > 0].copy()

    player_df["K_rate"] = player_df[so_col] / player_df[pa_col]
    player_df["HR_rate"] = player_df[hr_col] / player_df[pa_col]
    player_df["ISO"] = player_df[slg_col] - player_df[avg_col]
    player_df["AVG"] = player_df[avg_col]
    player_df["OPS"] = player_df[ops_col]

    player_df[BATTER_STAT_COLUMNS] = (player_df[BATTER_STAT_COLUMNS].replace([np.inf, -np.inf], np.nan))
    player_df = player_df.dropna(subset=BATTER_STAT_COLUMNS).copy()

    fallback_stats_base = player_df[BATTER_STAT_COLUMNS].mean().to_numpy(dtype=np.float32)
    fallback_stats = make_worse_fallback_stats(fallback_stats_base)
    stat_map = {
        int(row[id_col]): row[BATTER_STAT_COLUMNS].to_numpy(dtype=np.float32)
        for _, row in player_df.iterrows()}

    print(
        f"loaded batting stats: {prev_year}, "
        f"file={excel_path.name}, "
        f"players={len(stat_map)}, "
        f"features={BATTER_STAT_COLUMNS}")

    return stat_map, fallback_stats

def get_batter_stat_vector_and_flag(batter_id, stat_map, fallback_stats):
    try:
        batter_id_int = int(batter_id)
    except Exception:
        return fallback_stats.astype(np.float32), 1

    if batter_id_int in stat_map:
        return stat_map[batter_id_int].astype(np.float32), 0

    return fallback_stats.astype(np.float32), 1

def finalize_dataset(X_all, C_all, y_all, meta_all):
    X_all = np.stack(X_all).astype(np.float32)
    C_all = np.stack(C_all).astype(np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    meta_df = pd.DataFrame(meta_all)
    return X_all, C_all, y_all, meta_df

def empty_variant_lists():
    return {key: {"X": [], "C": [], "y": [], "meta": []}
        for key in VARIANT_CONFIGS}

def append_sample(store, sample):
    seq, c_vector, target, meta = sample
    store["X"].append(seq)
    store["C"].append(c_vector)
    store["y"].append(target)
    store["meta"].append(meta)

def add_screening_survivor(screening, stage, n_pitches):
    screening[f"{stage}_pas"] += 1
    screening[f"{stage}_pitches"] += int(n_pitches)

def init_year_summary():
    return {
        "total_samples": 0,
        "pitchers": set(),
        "batter_side": Counter(),
        "pitch_type": Counter(),
        "count": Counter(),
        "result": Counter(),
    }

def classify_pa_result(last_pitch):
    desc = last_pitch.get("description", None)
    event = last_pitch.get("events", None)

    # With two strikes before the final pitch, a swinging strike ends the PA
    # as a strikeout even if the events field is missing.
    if desc in {"swinging_strike", "swinging_strike_blocked"}:
        return "strikeout"

    if pd.isna(event):
        return "unknown"

    event = str(event)

    hit_events = {
        "single": "single",
        "double": "double",
        "triple": "triple",
        "home_run": "home_run",
    }
    if event in hit_events:
        return hit_events[event]

    if event in {"strikeout", "strikeout_double_play"}:
        return "strikeout"

    out_events = {
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "fielders_choice_out",
        "sac_fly",
        "sac_bunt",
        "double_play",
        "triple_play",
    }
    if event in out_events:
        return "out"

    return f"other:{event}"

def update_year_summary(summary, g):
    g = g.sort_values("pitch_number")
    last_pitch = g.iloc[-1]

    # Plate-appearance level statistics.
    summary["total_samples"] += 1

    pitcher = last_pitch.get("pitcher", None)
    if pd.notna(pitcher):
        try:
            pitcher = int(pitcher)
        except Exception:
            pitcher = str(pitcher)
        summary["pitchers"].add(pitcher)

    stand = last_pitch.get("stand", None)
    if pd.notna(stand):
        summary["batter_side"][str(stand)] += 1
    else:
        summary["batter_side"]["unknown"] += 1

    balls = last_pitch.get("balls", None)
    strikes = last_pitch.get("strikes", None)
    if pd.notna(balls) and pd.notna(strikes):
        summary["count"][f"{int(balls)}-{int(strikes)}"] += 1
    else:
        summary["count"]["unknown"] += 1

    summary["result"][classify_pa_result(last_pitch)] += 1

    # Pitch-type counts use every pitch in each accepted plate appearance,
    # not only the final pitch and not only the last MAX_PITCHES pitches.
    if "pitch_type" in g.columns:
        pitch_types = g["pitch_type"].fillna("unknown")
        for pitch_type in pitch_types:
            summary["pitch_type"][str(pitch_type)] += 1
    else:
        summary["pitch_type"]["unknown"] += len(g)

def save_year_summary(year, summary, output_dir):
    rows = []

    def add_row(metric, category, count):
        rows.append({
            "year": int(year),
            "metric": metric,
            "category": category,
            "count": int(count),
        })

    add_row("total_samples", "all", summary["total_samples"])
    add_row("pitchers", "all", len(summary["pitchers"]))

    # Always write R and L rows even when one side has zero samples.
    for side in ["R", "L"]:
        add_row("batter_side", side, summary["batter_side"].get(side, 0))
    for side, count in sorted(summary["batter_side"].items()):
        if side not in {"R", "L"}:
            add_row("batter_side", side, count)

    for pitch_type, count in sorted(summary["pitch_type"].items()):
        add_row("pitch_type", pitch_type, count)

    # The current sample definition requires two strikes before the last pitch,
    # so these are the expected terminal counts.
    for count_label in ["0-2", "1-2", "2-2", "3-2"]:
        add_row("count", count_label, summary["count"].get(count_label, 0))
    for count_label, count in sorted(summary["count"].items()):
        if count_label not in {"0-2", "1-2", "2-2", "3-2"}:
            add_row("count", count_label, count)

    result_order = [
        "strikeout", "out", "single", "double",
        "triple", "home_run", "unknown"
    ]
    for result in result_order:
        if result == "unknown" or result in summary["result"]:
            add_row("result", result, summary["result"].get(result, 0))
    for result, count in sorted(summary["result"].items()):
        if result not in result_order:
            add_row("result", result, count)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meta_{year}_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path

def init_screening_record(csv_path, year, raw_pitches=0, raw_pas=0):
    screening = {"year": int(year),
        "source_file": csv_path.name,
        "file_status": "ok",
        "missing_columns": ""}
    for stage, _, _ in SCREENING_STAGE_DEFS:
        screening[f"{stage}_pas"] = 0
        screening[f"{stage}_pitches"] = 0

    screening["raw_pitches"] = int(raw_pitches)
    screening["raw_pas"] = int(raw_pas)
    return screening

def get_description_before_last(g, offset):
    if len(g) < offset + 1:
        return None
    return g.iloc[-(offset + 1)]["description"]

def make_one_sample(g, csv_path, year, stat_map, fallback_stats, screening):
    g = g.sort_values("pitch_number")
    last_pitch = g.iloc[-1]
    n_pitches_in_pa = len(g)

    if n_pitches_in_pa < 3:
        return None

    if last_pitch["strikes"] != 2:
        return None    

    if last_pitch["strikes"] != 2:
        return None
    add_screening_survivor(screening, "last_pitch_two_strikes", n_pitches_in_pa)

    target = get_target(last_pitch)
    if target is None:
        return None
    add_screening_survivor(screening, "valid_target", n_pitches_in_pa)

    seq_df = g[PITCH_FEATURE_COLUMNS]
    if seq_df.isna().any().any():
        return None
    add_screening_survivor(screening, "complete_pitch_features", n_pitches_in_pa)

    context_values = last_pitch[CONTEXT_STATIC_COLUMNS]
    if context_values.isna().any():
        return None
    add_screening_survivor(screening, "complete_context", n_pitches_in_pa)

    seq = seq_df.to_numpy(dtype=np.float32)
    seq = pad_sequence(seq, MAX_PITCHES)

    batter_stats, used_fallback = get_batter_stat_vector_and_flag(
        batter_id=last_pitch["batter"],
        stat_map=stat_map,
        fallback_stats=fallback_stats)
    context_vector = context_values.to_numpy(dtype=np.float32)

    c_vector = np.concatenate([
        batter_stats.astype(np.float32),
        np.array([used_fallback], dtype=np.float32),
        context_vector.astype(np.float32)]).astype(np.float32)

    descriptions_before_last = {
        offset: get_description_before_last(g, offset)
        for offset in BALL_OFFSETS}

    meta = {
        "year": year,
        "prev_batting_year": year - 1,
        "source_file": csv_path.name,
        "game_pk": last_pitch["game_pk"],
        "game_date": last_pitch["game_date"] if "game_date" in last_pitch.index else None,
        "at_bat_number": last_pitch["at_bat_number"],
        "pitcher": last_pitch["pitcher"],
        "batter": last_pitch["batter"],
        "pitch_count_in_pa": n_pitches_in_pa,
        "prev_description": descriptions_before_last[1],
        "last_description": last_pitch["description"],
        "used_batter_stats_fallback": used_fallback,
        "target": target}

    for offset, desc in descriptions_before_last.items():
        meta[f"description_{offset}_before_last"] = desc

    for col, value in zip(BATTER_STAT_COLUMNS, batter_stats):
        meta[col] = float(value)

    for col, value in zip(CONTEXT_STATIC_COLUMNS, context_vector):
        meta[f"last_{col}"] = float(value)

    return seq, c_vector, target, meta

def build_dataset_from_csv(csv_path, year, stat_map, fallback_stats, year_summary):
    df = pd.read_csv(csv_path)

    pa_cols = ["game_pk", "at_bat_number", "pitcher", "batter"]
    raw_pas = 0
    if all(col in df.columns for col in pa_cols):
        raw_pas = df[pa_cols].drop_duplicates().shape[0]

    screening = init_screening_record(
        csv_path=csv_path,
        year=year,
        raw_pitches=len(df),
        raw_pas=raw_pas)
    variants = empty_variant_lists()

    preprocess_required_cols = [
        "plate_z", "sz_bot", "sz_top", "stand",
        "on_1b", "on_2b", "on_3b", "inning_topbot"]

    missing_preprocess_cols = [
        col for col in preprocess_required_cols if col not in df.columns]

    if missing_preprocess_cols:
        screening["file_status"] = "skipped_missing_columns"
        screening["missing_columns"] = ",".join(missing_preprocess_cols)
        print(f"skip {csv_path.name}: missing columns = {missing_preprocess_cols}")
        return variants, screening

    df = preprocess_pitch_df(df)

    numeric_cols = (PITCH_FEATURE_COLUMNS
        + CONTEXT_STATIC_COLUMNS
        + [TARGET_COLUMN, "pitch_number", "strikes", "batter"])

    numeric_cols = list(dict.fromkeys(numeric_cols))
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = (pa_cols
        + PITCH_FEATURE_COLUMNS
        + CONTEXT_STATIC_COLUMNS
        + [TARGET_COLUMN, "description", "pitch_number", "strikes"])

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
       screening["file_status"] = "skipped_missing_columns"
       screening["missing_columns"] = ",".join(missing_cols)
       return variants, screening

    df = df.sort_values(pa_cols + ["pitch_number"])

    for _, g in df.groupby(pa_cols, sort=False):
        sample = make_one_sample(
            g=g, csv_path=csv_path,
            year=year, stat_map=stat_map,
            fallback_stats=fallback_stats,
            screening=screening)

        if sample is None:
           continue

        append_sample(variants["normal"], sample)
        update_year_summary(year_summary, g)
        meta = sample[3]
        n_pitches_in_pa = meta["pitch_count_in_pa"]

        add_screening_survivor(screening,
            "final_normal",
            n_pitches_in_pa)

        for offset in BALL_OFFSETS:
            variant_key = f"ball_{offset}_before"
            desc = meta[f"description_{offset}_before_last"]
            if desc in BALL_DESCRIPTIONS:
                append_sample(variants[variant_key], sample)
                add_screening_survivor(screening, variant_key, n_pitches_in_pa)

    return variants, screening

def build_dataset(years):
    combined = empty_variant_lists()
    batting_cache = {}
    screening_records = []
    year_summaries = {year: init_year_summary() for year in years}

    for year in years:
        prev_year = year - 1
        if prev_year not in batting_cache:
           batting_cache[prev_year] = load_batter_stat_info(prev_year)

        stat_map, fallback_stats = batting_cache[prev_year]
        year_dir = DATA_ROOT / str(year)
        csv_files = sorted(year_dir.glob("*.csv"))

        print(f"\n{year}: {len(csv_files)} files")
        for csv_path in tqdm(csv_files):
            file_variants, screening = build_dataset_from_csv(
                csv_path=csv_path,
                year=year,
                stat_map=stat_map,
                fallback_stats=fallback_stats,
                year_summary=year_summaries[year])
            screening_records.append(screening)

            for variant_key in VARIANT_CONFIGS:
                part = file_variants[variant_key]
                combined[variant_key]["X"].extend(part["X"])
                combined[variant_key]["C"].extend(part["C"])
                combined[variant_key]["y"].extend(part["y"])
                combined[variant_key]["meta"].extend(part["meta"])

    results = {
        variant_key: finalize_dataset(
            combined[variant_key]["X"],
            combined[variant_key]["C"],
            combined[variant_key]["y"],
            combined[variant_key]["meta"])
        for variant_key in VARIANT_CONFIGS
    }

    screening_df = pd.DataFrame(screening_records)
    return results, screening_df, year_summaries

def make_screening_summary(screening_df, years=None, scope="all"):
    if years is None:
       df = screening_df.copy()
       years_label = "all"
    else:
       years = list(years)
       df = screening_df[screening_df["year"].isin(years)].copy()
       years_label = ",".join(map(str, years))

    rows = []
    stage_totals = {}
    for stage, label, parent_stage in SCREENING_STAGE_DEFS:
        pitches = int(df[f"{stage}_pitches"].sum())
        pas = int(df[f"{stage}_pas"].sum())
        stage_totals[stage] = {"pitches": pitches, "pas": pas}

        raw_pitches = stage_totals.get("raw", {}).get("pitches", 0)
        raw_pas = stage_totals.get("raw", {}).get("pas", 0)

        if parent_stage is None:
            parent_pitches = np.nan
            parent_pas = np.nan
            excluded_pitches = np.nan
            excluded_pas = np.nan
            pitch_retention_parent = np.nan
            pa_retention_parent = np.nan
        else:
            parent_pitches = stage_totals[parent_stage]["pitches"]
            parent_pas = stage_totals[parent_stage]["pas"]
            excluded_pitches = parent_pitches - pitches
            excluded_pas = parent_pas - pas
            pitch_retention_parent = (100.0 * pitches / parent_pitches)
            pa_retention_parent = (100.0 * pas / parent_pas)

        rows.append({
            "scope": scope,
            "years": years_label,
            "stage": stage,
            "stage_label": label,
            "parent_stage": parent_stage,
            "pitches": pitches,
            "plate_appearances": pas,
            "excluded_pitches_from_parent": excluded_pitches,
            "excluded_pa_from_parent": excluded_pas,
            "pitch_retention_vs_parent_pct": pitch_retention_parent,
            "pa_retention_vs_parent_pct": pa_retention_parent,
            "pitch_retention_vs_raw_pct": (100.0 * pitches / raw_pitches),
            "pa_retention_vs_raw_pct": (100.0 * pas / raw_pas)})

    return pd.DataFrame(rows)

def append_year_screening_to_sample_info(year, screening_df):
    """Append major sample-selection counts to the year's sheet."""

    if not SAMPLE_INFO_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_INFO_PATH} was not found. "
            "Run the data downloader first so the workbook already contains "
            "the Target pitcher selection row."
        )

    sheet_name = str(year)
    try:
        log_df = pd.read_excel(SAMPLE_INFO_PATH, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(
            f"Sheet {sheet_name} was not found in {SAMPLE_INFO_PATH}."
        ) from exc

    missing_sample_cols = [
        col for col in SAMPLE_COLUMNS if col not in log_df.columns
    ]
    if missing_sample_cols:
        raise ValueError(
            f"{SAMPLE_INFO_PATH} sheet {sheet_name} is missing columns: "
            f"{missing_sample_cols}"
        )

    summary = make_screening_summary(
        screening_df,
        years=[year],
        scope=f"year_{year}",
    )
    stage_info = {row["stage"]: row for _, row in summary.iterrows()}

    # Use the accepted normal sample as the two-strike outcome sample.
    # The finer internal screening steps are intentionally not written to Excel.
    raw_pitches = int(stage_info["raw"]["pitches"])
    two_strike_pitches = int(stage_info["final_normal"]["pitches"])

    new_rows = [
        {
            "process_order": 1,
            "process_name": "Two-strike outcome selection",
            "input_sample_size": raw_pitches,
            "output_sample_size": two_strike_pitches,
            "excluded_sample_size": raw_pitches - two_strike_pitches,
            "retention_rate": (
                two_strike_pitches / raw_pitches
                if raw_pitches > 0
                else np.nan
            ),
        }
    ]

    # Each ball-offset sample is an independent subset of the same
    # two-strike outcome sample, not a sequential filter of the previous row.
    for i, offset in enumerate(BALL_OFFSETS, start=2):
        output_pitches = int(stage_info[f"ball_{offset}_before"]["pitches"])
        pitch_word = "pitch" if offset == 1 else "pitches"
        new_rows.append({
            "process_order": i,
            "process_name": f"Ball {offset} {pitch_word} before final pitch",
            "input_sample_size": two_strike_pitches,
            "output_sample_size": output_pitches,
            "excluded_sample_size": two_strike_pitches - output_pitches,
            "retention_rate": (
                output_pitches / two_strike_pitches
                if two_strike_pitches > 0
                else np.nan
            ),
        })

    new_df = pd.DataFrame(new_rows, columns=SAMPLE_COLUMNS)

    # Remove any rows previously written by dataset-generator versions while
    # preserving the downloader's Target pitcher selection row.
    generated_names = {
        "Dataset generation",
        "Two-strike outcome selection",
        *{
            f"Ball {offset} {'pitch' if offset == 1 else 'pitches'} before final pitch"
            for offset in BALL_OFFSETS
        },
        *{
            label for stage, label, _ in SCREENING_STAGE_DEFS if stage != "raw"
        },
        *{
            f"Sensitivity dataset: ball {offset} pitch(es) before last"
            for offset in BALL_OFFSETS
        },
    }
    log_df = log_df[~log_df["process_name"].isin(generated_names)].copy()

    log_df = pd.concat([log_df, new_df], ignore_index=True)
    log_df = log_df[SAMPLE_COLUMNS].sort_values(
        ["process_order", "process_name"]
    ).reset_index(drop=True)

    with pd.ExcelWriter(
        SAMPLE_INFO_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        log_df.to_excel(writer, sheet_name=sheet_name, index=False)

def subset_dataset_by_years(dataset, years):
    X_all, C_all, y_all, meta_all = dataset
    mask = meta_all["year"].isin(years).to_numpy()
    return (
        X_all[mask],
        C_all[mask],
        y_all[mask],
        meta_all.loc[mask].reset_index(drop=True))

def save_variant_split(variant_key, train_data, test_data, train_year_tag):
    config = VARIANT_CONFIGS[variant_key]
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config["file_stem"]

    X_train, C_train, y_train, meta_train = train_data
    X_test, C_test, y_test, meta_test = test_data

    extra_npz = {}
    if config["ball_offset"] is not None:
        extra_npz = {
            "ball_descriptions": np.array(sorted(BALL_DESCRIPTIONS)),
            "ball_offset_before_last": np.array([config["ball_offset"]], dtype=np.int32)}

    np.savez_compressed(
        output_dir / f"dataset_{stem}_{train_year_tag}.npz",
        X_train=X_train,
        C_train=C_train,
        y_train=y_train,
        X_test=X_test,
        C_test=C_test,
        y_test=y_test,
        pitch_feature_columns=np.array(PITCH_FEATURE_COLUMNS),
        c_feature_columns=np.array(C_FEATURE_COLUMNS),
        batter_stat_columns=np.array(BATTER_STAT_COLUMNS),
        context_static_columns=np.array(CONTEXT_STATIC_COLUMNS),
        bad_fallback_multiplier_keys=np.array(
            list(BAD_FALLBACK_MULTIPLIER.keys())),
        bad_fallback_multiplier_values=np.array(
            list(BAD_FALLBACK_MULTIPLIER.values())),
        **extra_npz)

    meta_train.to_csv(
        output_dir / f"meta_train_{stem}_{train_year_tag}.csv",
        index=False)
    meta_test.to_csv(
        output_dir / f"meta_test_{stem}_{train_year_tag}.csv",
        index=False)

ALL_YEARS = sorted({
    year
    for train_years, test_years in YEAR_SPLITS
    for year in train_years + test_years})

# Build all dataset variants once, together with file-level screening counts
# and annual metadata summaries for accepted normal samples.
all_variants, screening_by_file, year_summaries = build_dataset(ALL_YEARS)

# Save annual metadata summaries.
META_SUMMARY_OUTPUT_DIR = Path("./dataset/preceding")
for year in ALL_YEARS:
    summary_path = save_year_summary(
        year=year,
        summary=year_summaries[year],
        output_dir=META_SUMMARY_OUTPUT_DIR)
    print(f"Saved annual metadata summary: {summary_path}")

# Append year-by-year major dataset-generation counts to sample_process_info.xlsx.
for year in ALL_YEARS:
    append_year_screening_to_sample_info(
        year=year,
        screening_df=screening_by_file,
    )
    print(f"Updated sample process info: {SAMPLE_INFO_PATH}, sheet={year}")

# Generate expanding-window train/test files for every dataset variant.
for TRAIN_YEARS, TEST_YEARS in YEAR_SPLITS:
    train_year_tag = (
        str(TRAIN_YEARS[0])
        if len(TRAIN_YEARS) == 1
        else f"{TRAIN_YEARS[0]}_{str(TRAIN_YEARS[-1])[-2:]}")

    print(
        f"\n===== dataset shapes: train={TRAIN_YEARS}, "
        f"test={TEST_YEARS} =====")

    for variant_key, config in VARIANT_CONFIGS.items():
        train_data = subset_dataset_by_years(
            all_variants[variant_key], TRAIN_YEARS)

        test_data = subset_dataset_by_years(
            all_variants[variant_key], TEST_YEARS)

        X_train, C_train, y_train, meta_train = train_data
        X_test, C_test, y_test, meta_test = test_data

        print(f"\n--- {variant_key}: {config['label']} ---")
        print("X_train:", X_train.shape)
        print("C_train:", C_train.shape)
        print("y_train:", y_train.shape)
        print("X_test :", X_test.shape)
        print("C_test :", C_test.shape)
        print("y_test :", y_test.shape)

        save_variant_split(
            variant_key=variant_key,
            train_data=train_data,
            test_data=test_data,
            train_year_tag=train_year_tag)

    print(f"\nSaved datasets: {train_year_tag}")
