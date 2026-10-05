from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Paths and analysis settings
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

SENSITIVITY_ROOT = PROJECT_ROOT / "Sensitivity_analysis"
SUMMARY_DIR = SENSITIVITY_ROOT / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LAST_ROOT = PROJECT_ROOT / "dataset/last"
DATASET_PRECEDING_ROOT = PROJECT_ROOT / "dataset/preceding"

PLATE_HALF_WIDTH_FT = 8.5 / 12.0
INZONE_X_EDGES = np.linspace(-PLATE_HALF_WIDTH_FT, PLATE_HALF_WIDTH_FT, 4)
INZONE_Z_EDGES = np.linspace(0.0, 1.0, 4)

YEAR_CONFIGS = [
    {"test_year": 2022, "train_tag": "2018_19"},
    {"test_year": 2023, "train_tag": "2018_22"},
    {"test_year": 2024, "train_tag": "2018_23"},
    {"test_year": 2025, "train_tag": "2018_24"},
]

ANALYSIS_CONFIGS = [
    {
        "simulation_condition": "sim1_sim",
        "analysis_type": "sim1",
        "target_condition": "final_pitch",
        "offset": 0,
    },
    {
        "simulation_condition": "sim1_act",
        "analysis_type": "sim1",
        "target_condition": "final_pitch",
        "offset": 0,
    },
    *[
        {
            "simulation_condition": "sim2_sim",
            "analysis_type": "sim2",
            "target_condition": f"preceding_{offset}",
            "offset": offset,
        }
        for offset in range(1, 6)
    ],
    *[
        {
            "simulation_condition": "sim2_act",
            "analysis_type": "sim2",
            "target_condition": f"preceding_{offset}",
            "offset": offset,
        }
        for offset in range(1, 6)
    ],
]

COMPARISON_SCOPES = [
    "all_candidates",
    "same_course_pitch_type_change",
    "same_pitch_type_course_change",
]


# =============================================================================
# Helpers
# =============================================================================
def meta_path_for_condition(analysis_type, train_tag, offset):
    if analysis_type == "sim1":
        return DATASET_LAST_ROOT / f"meta_test_last_{train_tag}.csv"

    return (
        DATASET_PRECEDING_ROOT
        / f"meta_test_preceding_{offset}_{train_tag}.csv"
    )


def result_dir_for_condition(simulation_condition, year, analysis_type, offset):
    base_dir = SENSITIVITY_ROOT / simulation_condition / str(year)

    if analysis_type == "sim1":
        return base_dir

    return base_dir / f"preceding_{offset}"


def assign_inzone_grid(plate_x, plate_z_norm):
    """
    Return the 3x3 strike-zone grid used by Simulation 1.

    Returns
    -------
    tuple[int, int] | None
        (grid_x_index, grid_z_index), or None when the original pitch is
        geometrically outside the 3x3 strike-zone grid.
    """
    if not (np.isfinite(plate_x) and np.isfinite(plate_z_norm)):
        return None

    if not (
        -PLATE_HALF_WIDTH_FT <= float(plate_x) <= PLATE_HALF_WIDTH_FT
        and 0.0 <= float(plate_z_norm) <= 1.0
    ):
        return None

    grid_x_index = int(
        np.searchsorted(INZONE_X_EDGES, float(plate_x), side="right") - 1
    )
    grid_z_index = int(
        np.searchsorted(INZONE_Z_EDGES, float(plate_z_norm), side="right") - 1
    )

    grid_x_index = int(np.clip(grid_x_index, 0, 2))
    grid_z_index = int(np.clip(grid_z_index, 0, 2))

    return grid_x_index, grid_z_index


def assign_outzone_region(plate_x, plate_z_norm):
    """
    Return the 8-region outside-zone label used by Simulation 2.

    Returns None when the pitch center is geometrically inside the rectangular
    strike zone or when the location is missing.
    """
    if not (np.isfinite(plate_x) and np.isfinite(plate_z_norm)):
        return None

    x = float(plate_x)
    z = float(plate_z_norm)

    inside_x = -PLATE_HALF_WIDTH_FT <= x <= PLATE_HALF_WIDTH_FT
    inside_z = 0.0 <= z <= 1.0

    left = x < -PLATE_HALF_WIDTH_FT
    right = x > PLATE_HALF_WIDTH_FT
    down = z < 0.0
    up = z > 1.0

    if left and inside_z:
        return "left"
    if right and inside_z:
        return "right"
    if inside_x and down:
        return "down"
    if inside_x and up:
        return "up"
    if left and down:
        return "down_left"
    if right and down:
        return "down_right"
    if left and up:
        return "up_left"
    if right and up:
        return "up_right"

    return None


def normalize_pitch_type(value):
    if pd.isna(value):
        return None
    return str(value)


def original_row_and_course(result_df, analysis_type):
    if analysis_type == "sim1":
        original_rows = result_df[result_df["row_type"] == "original"]
    else:
        original_rows = result_df[result_df["row_type"] == "original_target"]

    if len(original_rows) != 1:
        raise ValueError(
            f"Expected exactly one original row, found {len(original_rows)}"
        )

    original_row = original_rows.iloc[0]

    if analysis_type == "sim1":
        grid = assign_inzone_grid(
            pd.to_numeric(original_row.get("plate_x"), errors="coerce"),
            pd.to_numeric(original_row.get("plate_z_norm"), errors="coerce"),
        )
        original_course = None if grid is None else f"grid_{grid[0]}_{grid[1]}"
    else:
        original_course = assign_outzone_region(
            pd.to_numeric(original_row.get("plate_x"), errors="coerce"),
            pd.to_numeric(original_row.get("plate_z_norm"), errors="coerce"),
        )

    return original_row, original_course


def add_candidate_course(candidate_df, analysis_type):
    candidate_df = candidate_df.copy()

    if analysis_type == "sim1":
        grid_x = pd.to_numeric(
            candidate_df["grid_x_index"], errors="coerce"
        ).astype("Int64")
        grid_z = pd.to_numeric(
            candidate_df["grid_z_index"], errors="coerce"
        ).astype("Int64")

        candidate_df["candidate_course"] = [
            (
                f"grid_{int(x)}_{int(z)}"
                if pd.notna(x) and pd.notna(z)
                else None
            )
            for x, z in zip(grid_x, grid_z)
        ]
    else:
        candidate_df["candidate_course"] = candidate_df["region"].astype(
            "string"
        )
        candidate_df.loc[
            candidate_df["candidate_course"].isna(),
            "candidate_course",
        ] = None

    return candidate_df


def select_candidates(
    candidate_df,
    comparison_scope,
    original_pitch_type,
    original_course,
):
    if comparison_scope == "all_candidates":
        return candidate_df

    if comparison_scope == "same_course_pitch_type_change":
        # "Same course, pitch type only changed":
        # keep the same grid/region, but exclude the original pitch type.
        if original_course is None:
            return candidate_df.iloc[0:0].copy()

        return candidate_df[
            (candidate_df["candidate_course"] == original_course)
            & (candidate_df["pitch_type_normalized"] != original_pitch_type)
        ].copy()

    if comparison_scope == "same_pitch_type_course_change":
        # "Same pitch type, course only changed":
        # keep the original pitch type and exclude the original grid/region.
        same_pitch = (
            candidate_df["pitch_type_normalized"] == original_pitch_type
        )

        if original_course is None:
            # The original pitch lies outside the candidate course domain.
            # Therefore every candidate course represents a location change.
            return candidate_df[same_pitch].copy()

        return candidate_df[
            same_pitch
            & (candidate_df["candidate_course"] != original_course)
        ].copy()

    raise ValueError(f"Unknown comparison scope: {comparison_scope}")


def summarize_one_sample(
    result_df,
    analysis_type,
    comparison_scope,
):
    original_row, original_course = original_row_and_course(
        result_df=result_df,
        analysis_type=analysis_type,
    )

    original_output = float(
        pd.to_numeric(original_row["model_output"], errors="raise")
    )
    original_pitch_type = normalize_pitch_type(original_row.get("pitch_type"))

    candidate_df = result_df[result_df["row_type"] == "simulation"].copy()
    if candidate_df.empty:
        return None

    candidate_df["model_output"] = pd.to_numeric(
        candidate_df["model_output"],
        errors="coerce",
    )
    candidate_df = candidate_df[
        np.isfinite(candidate_df["model_output"].to_numpy(dtype=float))
    ].copy()

    if candidate_df.empty:
        return None

    candidate_df["pitch_type_normalized"] = candidate_df["pitch_type"].map(
        normalize_pitch_type
    )
    candidate_df = add_candidate_course(candidate_df, analysis_type)

    selected_df = select_candidates(
        candidate_df=candidate_df,
        comparison_scope=comparison_scope,
        original_pitch_type=original_pitch_type,
        original_course=original_course,
    )

    if selected_df.empty:
        return None

    candidate_outputs = selected_df["model_output"].to_numpy(dtype=float)
    deltas = candidate_outputs - original_output

    max_delta = float(np.max(deltas))
    min_delta = float(np.min(deltas))

    return {
        "original_model_output": original_output,
        "original_pitch_type": original_pitch_type,
        "original_course": original_course,
        "n_candidates": int(len(selected_df)),
        "max_candidate_model_output": float(np.max(candidate_outputs)),
        "min_candidate_model_output": float(np.min(candidate_outputs)),
        # Neutral signed changes relative to the original prediction.
        "max_increase_delta": max_delta,
        "max_decrease_delta": min_delta,
        # Positive magnitudes are convenient for interpreting the maximum
        # possible upward/downward change separately.
        "max_increase_magnitude": max(0.0, max_delta),
        "max_decrease_magnitude": max(0.0, -min_delta),
        # Total span between the highest and lowest simulated probabilities.
        "extreme_output_range": float(
            np.max(candidate_outputs) - np.min(candidate_outputs)
        ),
    }


def build_pitcher_identity(meta_row):
    source_file = str(meta_row.get("source_file", ""))
    player_name = Path(source_file).stem if source_file else ""

    pitcher_value = meta_row.get("pitcher", np.nan)
    if pd.isna(pitcher_value):
        pitcher_id = ""
        pitcher_key = player_name
    else:
        try:
            pitcher_id = str(int(float(pitcher_value)))
        except (TypeError, ValueError):
            pitcher_id = str(pitcher_value)
        pitcher_key = pitcher_id

    return pitcher_key, pitcher_id, player_name, source_file


def collect_sample_level_rows():
    sample_rows = []

    for analysis_config in ANALYSIS_CONFIGS:
        simulation_condition = analysis_config["simulation_condition"]
        analysis_type = analysis_config["analysis_type"]
        target_condition = analysis_config["target_condition"]
        offset = int(analysis_config["offset"])

        for year_config in YEAR_CONFIGS:
            year = int(year_config["test_year"])
            train_tag = year_config["train_tag"]

            meta_path = meta_path_for_condition(
                analysis_type=analysis_type,
                train_tag=train_tag,
                offset=offset,
            )
            result_dir = result_dir_for_condition(
                simulation_condition=simulation_condition,
                year=year,
                analysis_type=analysis_type,
                offset=offset,
            )

            if not meta_path.is_file():
                print(f"[SKIP] Metadata not found: {meta_path}")
                continue
            if not result_dir.is_dir():
                print(f"[SKIP] Result directory not found: {result_dir}")
                continue

            meta_test = pd.read_csv(meta_path)
            result_files = sorted(result_dir.glob("*.csv"))

            print(
                f"\n[{simulation_condition}] "
                f"year={year}, target={target_condition}, "
                f"files={len(result_files)}"
            )

            for result_path in result_files:
                try:
                    sample_index = int(result_path.stem)
                except ValueError:
                    # Ignore non-numbered CSV files if any are present.
                    continue

                if sample_index < 0 or sample_index >= len(meta_test):
                    print(
                        f"[SKIP] Sample index outside metadata range: "
                        f"{result_path}"
                    )
                    continue

                meta_row = meta_test.iloc[sample_index]
                (
                    pitcher_key,
                    pitcher_id,
                    player_name,
                    source_file,
                ) = build_pitcher_identity(meta_row)

                try:
                    result_df = pd.read_csv(result_path)
                except Exception as exc:
                    print(f"[SKIP] Could not read {result_path}: {exc!r}")
                    continue

                for comparison_scope in COMPARISON_SCOPES:
                    try:
                        summary = summarize_one_sample(
                            result_df=result_df,
                            analysis_type=analysis_type,
                            comparison_scope=comparison_scope,
                        )
                    except Exception as exc:
                        print(
                            f"[SKIP] Failed to summarize {result_path} "
                            f"scope={comparison_scope}: {exc!r}"
                        )
                        continue

                    if summary is None:
                        continue

                    sample_rows.append(
                        {
                            "simulation_condition": simulation_condition,
                            "year": year,
                            "target_condition": target_condition,
                            "ball_offset_before_last": offset,
                            "comparison_scope": comparison_scope,
                            "sample_index": sample_index,
                            "pitcher_key": pitcher_key,
                            "pitcher_id": pitcher_id,
                            "player_name": player_name,
                            "source_file": source_file,
                            **summary,
                        }
                    )

    return pd.DataFrame(sample_rows)


def aggregate_by_pitcher(sample_df):
    if sample_df.empty:
        return pd.DataFrame()

    group_columns = [
        "simulation_condition",
        "year",
        "target_condition",
        "ball_offset_before_last",
        "comparison_scope",
        "pitcher_key",
        "pitcher_id",
        "player_name",
    ]

    pitcher_df = (
        sample_df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n_pitches=("sample_index", "size"),
            mean_candidates_per_pitch=("n_candidates", "mean"),
            mean_original_model_output=("original_model_output", "mean"),
            mean_max_increase_delta=("max_increase_delta", "mean"),
            mean_max_decrease_delta=("max_decrease_delta", "mean"),
            mean_max_increase_magnitude=("max_increase_magnitude", "mean"),
            mean_max_decrease_magnitude=("max_decrease_magnitude", "mean"),
            mean_extreme_output_range=("extreme_output_range", "mean"),
        )
    )

    return pitcher_df


def aggregate_across_pitchers(pitcher_df):
    if pitcher_df.empty:
        return pd.DataFrame()

    group_columns = [
        "simulation_condition",
        "year",
        "target_condition",
        "ball_offset_before_last",
        "comparison_scope",
    ]

    # IMPORTANT:
    # These are means of pitcher-level means. Every pitcher receives equal
    # weight regardless of how many pitches that pitcher contributed.
    overall_df = (
        pitcher_df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n_pitchers=("pitcher_key", "nunique"),
            total_pitches_used=("n_pitches", "sum"),
            mean_pitches_per_pitcher=("n_pitches", "mean"),
            mean_original_model_output=(
                "mean_original_model_output",
                "mean",
            ),
            mean_max_increase_delta=(
                "mean_max_increase_delta",
                "mean",
            ),
            mean_max_decrease_delta=(
                "mean_max_decrease_delta",
                "mean",
            ),
            mean_max_increase_magnitude=(
                "mean_max_increase_magnitude",
                "mean",
            ),
            mean_max_decrease_magnitude=(
                "mean_max_decrease_magnitude",
                "mean",
            ),
            mean_extreme_output_range=(
                "mean_extreme_output_range",
                "mean",
            ),
        )
    )

    return overall_df


# =============================================================================
# Run summary
# =============================================================================
sample_summary_df = collect_sample_level_rows()

if sample_summary_df.empty:
    raise RuntimeError(
        "No sensitivity-analysis result files were summarized. "
        "Check Sensitivity_analysis/ and the dataset metadata paths."
    )

pitcher_summary_df = aggregate_by_pitcher(sample_summary_df)
overall_summary_df = aggregate_across_pitchers(pitcher_summary_df)

pitcher_output_path = SUMMARY_DIR / "sensitivity_summary_by_pitcher.csv"
overall_output_path = SUMMARY_DIR / "sensitivity_summary_overall.csv"

pitcher_summary_df.to_csv(pitcher_output_path, index=False)
overall_summary_df.to_csv(overall_output_path, index=False)

print("\n===== Sensitivity summary complete =====")
print(f"Pitcher-level summary: {pitcher_output_path}")
print(f"Overall summary      : {overall_output_path}")
print("\nOverall summary:")
print(overall_summary_df.to_string(index=False))
