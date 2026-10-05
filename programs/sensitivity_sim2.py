from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models

MAX_PITCHES = 6
FEATURE_COLUMNS = [
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z_norm",
]

PLATE_HALF_WIDTH_FT = 8.5 / 12.0
BALL_DESCRIPTIONS = {"ball", "blocked_ball"}

# Use one representative point at the center of each ball-zone region.
# The strike-zone center is 0.50. The low/high outside regions are
# represented by the midpoint of the original sampled values:
# (-0.10 + -0.20) / 2 = -0.15 and (1.10 + 1.20) / 2 = 1.15.
INSIDE_GRID = np.array([0.50])
LOW_OUTSIDE_GRID = np.array([-0.15])
HIGH_OUTSIDE_GRID = np.array([1.15])

# 8 regions: left, right, down, up,
# down_left, down_right, up_left, up_right.
REGIONS = {
    "left": {
        "x_norm_values": LOW_OUTSIDE_GRID,
        "z_norm_values": INSIDE_GRID,
    },
    "right": {
        "x_norm_values": HIGH_OUTSIDE_GRID,
        "z_norm_values": INSIDE_GRID,
    },
    "down": {
        "x_norm_values": INSIDE_GRID,
        "z_norm_values": LOW_OUTSIDE_GRID,
    },
    "up": {
        "x_norm_values": INSIDE_GRID,
        "z_norm_values": HIGH_OUTSIDE_GRID,
    },
    "down_left": {
        "x_norm_values": LOW_OUTSIDE_GRID,
        "z_norm_values": LOW_OUTSIDE_GRID,
    },
    "down_right": {
        "x_norm_values": HIGH_OUTSIDE_GRID,
        "z_norm_values": LOW_OUTSIDE_GRID,
    },
    "up_left": {
        "x_norm_values": LOW_OUTSIDE_GRID,
        "z_norm_values": HIGH_OUTSIDE_GRID,
    },
    "up_right": {
        "x_norm_values": HIGH_OUTSIDE_GRID,
        "z_norm_values": HIGH_OUTSIDE_GRID,
    },
}

@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class PaddingMaskLayer(layers.Layer):
    def call(self, inputs):
        is_padding = tf.reduce_all(tf.equal(inputs, 0.0), axis=-1)
        return tf.logical_not(is_padding)


@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(
            input_dim=seq_len,
            output_dim=d_model,
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization()
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x, valid_mask = inputs
        attn_mask = tf.cast(valid_mask[:, tf.newaxis, :], tf.bool)
        attn_output = self.attn(
            x,
            x,
            attention_mask=attn_mask,
            training=training,
        )
        x = self.norm1(x + self.dropout1(attn_output, training=training))

        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class MaskedMeanPooling(layers.Layer):
    def call(self, inputs):
        x, valid_mask = inputs
        mask = tf.cast(valid_mask, tf.float32)[:, :, tf.newaxis]
        sum_x = tf.reduce_sum(x * mask, axis=1)
        count = tf.reduce_sum(mask, axis=1)
        return sum_x / tf.maximum(count, 1.0)

def dataset_path_for_offset(offset):
    return DATASET_ROOT / f"dataset_preceding_{offset}_{TRAIN_YEAR_TAG}.npz"


def meta_test_path_for_offset(offset):
    return DATASET_ROOT / f"meta_test_preceding_{offset}_{TRAIN_YEAR_TAG}.csv"


def plate_x_norm_to_ft(x_norm):
    return (2.0 * x_norm - 1.0) * PLATE_HALF_WIDTH_FT


def transform_sequence_standardizer(X, mean, std):
    X_scaled = X.copy().astype(np.float32)
    flat = X_scaled.reshape(-1, X_scaled.shape[-1])
    non_padding_mask = ~np.all(flat == 0, axis=1)
    flat[non_padding_mask] = (flat[non_padding_mask] - mean) / std
    return X_scaled


def transform_tabular_standardizer(C, mean, std):
    return ((C.astype(np.float32) - mean) / std).astype(np.float32)


def predict_model_output(X_seq, C_context):
    X_seq = np.asarray(X_seq, dtype=np.float32)
    C_context = np.asarray(C_context, dtype=np.float32)

    X_scaled = transform_sequence_standardizer(X_seq, x_mean, x_std)
    C_scaled = transform_tabular_standardizer(C_context, c_mean, c_std)

    raw_predictions = model.predict(
        {
            "pitch_sequence": X_scaled,
            "context_features": C_scaled,
        },
        verbose=0,
    ).ravel()

    # Sensitivity analyses use the Platt-calibrated probability saved together
    # with the expanding-window model for the corresponding test year.
    return apply_saved_platt_scaling(raw_predictions)


def add_plate_z_norm(df):
    df = df.copy()
    df["plate_z_norm"] = (
        (df["plate_z"] - df["sz_bot"])
        / (df["sz_top"] - df["sz_bot"])
    )
    return df


def normalize_x_for_batter_view(df):
    df = df.copy()
    left_mask = df["stand"] == "L"
    for col in ["pfx_x", "plate_x"]:
        if col in df.columns:
            df.loc[left_mask, col] = -df.loc[left_mask, col]
    return df


@lru_cache(maxsize=32)
def load_pitch_type_table(source_file):
    path = PITCH_TYPE_ROOT / str(source_file)
    df = pd.read_csv(path)
    features = [
        "pitch_type",
        "pfx_x",
        "pfx_z",
        "effective_speed",
        "release_spin_rate",
        "spin_axis",
    ]
    for col in features[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=features).copy()


@lru_cache(maxsize=32)
def load_raw_source_file(source_file):
    path = RAW_DATA_ROOT / str(source_file)
    df = pd.read_csv(path)

    numeric_cols = [
        "game_pk",
        "at_bat_number",
        "pitcher",
        "batter",
        "pitch_number",
        "strikes",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "effective_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_plate_z_norm(df)
    df = normalize_x_for_batter_view(df)
    return df


def find_original_target_and_last_pitch(meta_row, ball_offset):
    source_file = meta_row["source_file"]
    raw_df = load_raw_source_file(source_file)

    cond = (
        (raw_df["game_pk"] == meta_row["game_pk"])
        & (raw_df["at_bat_number"] == meta_row["at_bat_number"])
        & (raw_df["pitcher"] == meta_row["pitcher"])
        & (raw_df["batter"] == meta_row["batter"])
    )

    g = raw_df.loc[cond].sort_values("pitch_number").copy()
    required_len = ball_offset + 1
    if len(g) < required_len:
        return None, None, (
            f"plate appearance has only {len(g)} pitches; "
            f"offset={ball_offset} requires at least {required_len}"
        )

    # offset=1 -> second-to-last pitch, offset=5 -> sixth-to-last pitch.
    target_pitch = g.iloc[-(ball_offset + 1)]
    last_pitch = g.iloc[-1]

    target_description = target_pitch.get("description", None)
    if target_description not in BALL_DESCRIPTIONS:
        return None, None, (
            f"target pitch at offset={ball_offset} is not ball/blocked_ball: "
            f"{target_description!r}"
        )

    return target_pitch, last_pitch, None


def make_simulation_rows(
    sample_index,
    meta_row,
    pitch_type_df,
    X_test,
    C_test,
    ball_offset,
):
    original_seq = X_test[sample_index].copy()
    original_context = C_test[sample_index].copy()

    target_pitch, last_pitch, reason = find_original_target_and_last_pitch(
        meta_row=meta_row,
        ball_offset=ball_offset,
    )
    if reason is not None:
        return None, reason

    # The sequence is right-aligned, so a pitch N before the final pitch is
    # always at index -(N+1), even when the plate appearance was padded.
    target_seq_index = -(ball_offset + 1)

    # A preceding_N dataset should guarantee that this position is non-padding.
    if np.all(original_seq[target_seq_index] == 0):
        return None, (
            f"target sequence position {target_seq_index} is padding "
            f"for offset={ball_offset}"
        )

    sz_top = target_pitch.get("sz_top", np.nan)
    sz_bot = target_pitch.get("sz_bot", np.nan)
    if not (np.isfinite(sz_top) and np.isfinite(sz_bot) and sz_top > sz_bot):
        return None, "invalid strike-zone height for target pitch"

    original_output = predict_model_output(
        original_seq.reshape(1, MAX_PITCHES, len(FEATURE_COLUMNS)),
        original_context.reshape(1, -1),
    )[0]

    original_target_feature = dict(
        zip(FEATURE_COLUMNS, original_seq[target_seq_index])
    )
    original_last_feature = dict(zip(FEATURE_COLUMNS, original_seq[-1]))

    common = {
        "ball_offset_before_last": int(ball_offset),
        "target_sequence_index": int(target_seq_index),
    }
    rows = []

    rows.append({
        **common,
        "row_type": "original_target",
        "region": "original_target",
        "pitch_type": target_pitch.get("pitch_type", np.nan),
        "grid_x_index": np.nan,
        "grid_z_index": np.nan,
        "effective_speed": original_target_feature["effective_speed"],
        "release_spin_rate": original_target_feature["release_spin_rate"],
        "spin_axis": original_target_feature["spin_axis"],
        "pfx_x": original_target_feature["pfx_x"],
        "pfx_z": original_target_feature["pfx_z"],
        "plate_x": original_target_feature["plate_x"],
        "plate_x_norm": np.nan,
        "plate_z_norm": original_target_feature["plate_z_norm"],
        "plate_z": target_pitch.get("plate_z", np.nan),
        "sz_bot": target_pitch.get("sz_bot", np.nan),
        "sz_top": target_pitch.get("sz_top", np.nan),
        "model_output": original_output,
    })

    rows.append({
        **common,
        "row_type": "original_last",
        "region": "original_last",
        "pitch_type": last_pitch.get("pitch_type", np.nan),
        "grid_x_index": np.nan,
        "grid_z_index": np.nan,
        "effective_speed": original_last_feature["effective_speed"],
        "release_spin_rate": original_last_feature["release_spin_rate"],
        "spin_axis": original_last_feature["spin_axis"],
        "pfx_x": original_last_feature["pfx_x"],
        "pfx_z": original_last_feature["pfx_z"],
        "plate_x": original_last_feature["plate_x"],
        "plate_x_norm": np.nan,
        "plate_z_norm": original_last_feature["plate_z_norm"],
        "plate_z": last_pitch.get("plate_z", np.nan),
        "sz_bot": last_pitch.get("sz_bot", np.nan),
        "sz_top": last_pitch.get("sz_top", np.nan),
        "model_output": original_output,
    })

    candidate_seqs = []
    candidate_meta = []
    stand = target_pitch.get("stand", None)
    x_sign = -1.0 if stand == "L" else 1.0

    for _, pt in pitch_type_df.iterrows():
        pitch_type = pt["pitch_type"]

        for region, grid in REGIONS.items():
            x_norm_values = grid["x_norm_values"]
            z_norm_values = grid["z_norm_values"]

            for ix, plate_x_norm in enumerate(x_norm_values):
                for iz, plate_z_norm in enumerate(z_norm_values):
                    seq_cf = original_seq.copy()
                    plate_x = plate_x_norm_to_ft(plate_x_norm)
                    plate_z_actual = (
                        sz_bot + plate_z_norm * (sz_top - sz_bot)
                    )
                    cf_pfx_x = pt["pfx_x"] * x_sign

                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("effective_speed"),
                    ] = pt["effective_speed"]
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("release_spin_rate"),
                    ] = pt["release_spin_rate"]
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("spin_axis"),
                    ] = pt["spin_axis"]
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("pfx_x"),
                    ] = cf_pfx_x
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("pfx_z"),
                    ] = pt["pfx_z"]
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("plate_x"),
                    ] = plate_x
                    seq_cf[
                        target_seq_index,
                        FEATURE_COLUMNS.index("plate_z_norm"),
                    ] = plate_z_norm

                    candidate_seqs.append(seq_cf)
                    candidate_meta.append({
                        **common,
                        "row_type": "simulation",
                        "region": region,
                        "pitch_type": pitch_type,
                        "grid_x_index": ix,
                        "grid_z_index": iz,
                        "effective_speed": pt["effective_speed"],
                        "release_spin_rate": pt["release_spin_rate"],
                        "spin_axis": pt["spin_axis"],
                        "pfx_x": cf_pfx_x,
                        "pfx_z": pt["pfx_z"],
                        "plate_x": plate_x,
                        "plate_x_norm": plate_x_norm,
                        "plate_z_norm": plate_z_norm,
                        "plate_z": plate_z_actual,
                        "sz_bot": sz_bot,
                        "sz_top": sz_top,
                    })

    if candidate_seqs:
        candidate_seqs = np.stack(candidate_seqs).astype(np.float32)
        candidate_contexts = np.repeat(
            original_context.reshape(1, -1),
            len(candidate_seqs),
            axis=0,
        ).astype(np.float32)

        outputs = predict_model_output(candidate_seqs, candidate_contexts)
        for metadata, output in zip(candidate_meta, outputs):
            metadata["model_output"] = float(output)
            rows.append(metadata)

    return pd.DataFrame(rows), None


def load_condition_data(ball_offset):
    dataset_path = dataset_path_for_offset(ball_offset)
    meta_path = meta_test_path_for_offset(ball_offset)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        X_test = data["X_test"].astype(np.float32)
        C_test = data["C_test"].astype(np.float32)
        y_test = (
            data["y_test"].astype(np.float32)
            if "y_test" in data.files
            else None
        )
        dataset_feature_columns = [
            str(value) for value in data["pitch_feature_columns"].tolist()
        ]
        stored_offsets = (
            data["ball_offset_before_last"].astype(int).ravel().tolist()
            if "ball_offset_before_last" in data.files
            else []
        )

    if dataset_feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Pitch-feature schema mismatch.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Dataset : {dataset_feature_columns}"
        )

    if stored_offsets and stored_offsets != [ball_offset]:
        raise ValueError(
            f"Dataset offset metadata {stored_offsets} does not match "
            f"requested offset={ball_offset}"
        )

    meta_test = pd.read_csv(meta_path)
    if len(meta_test) != len(X_test):
        raise ValueError(
            f"Length mismatch for offset={ball_offset}: "
            f"meta={len(meta_test)}, X_test={len(X_test)}"
        )

    return X_test, C_test, y_test, meta_test, dataset_path, meta_path


def run_condition(ball_offset, year_result_dir):
    (
        X_test,
        C_test,
        y_test,
        meta_test,
        dataset_path,
        meta_path,
    ) = load_condition_data(ball_offset)

    condition_name = f"preceding_{ball_offset}"
    condition_dir = year_result_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 100)
    print(f"Condition : {condition_name}")
    print(f"Dataset   : {dataset_path}")
    print(f"Meta test : {meta_path}")
    print(f"Samples   : {len(meta_test)}")

    n_saved = 0
    n_failed = 0

    for sample_index, meta_row in tqdm(
        meta_test.iterrows(),
        total=len(meta_test),
        desc=condition_name,
    ):
        source_file = Path(str(meta_row["source_file"])).name

        try:
            pitch_type_df = load_pitch_type_table(source_file)
            if pitch_type_df is None or pitch_type_df.empty:
                raise ValueError(f"No pitch-type candidates for {source_file}")

            result_df, reason = make_simulation_rows(
                sample_index=sample_index,
                meta_row=meta_row,
                pitch_type_df=pitch_type_df,
                X_test=X_test,
                C_test=C_test,
                ball_offset=ball_offset,
            )
            if reason is not None:
                raise ValueError(reason)

            result_df.to_csv(
                condition_dir / f"{sample_index:06d}.csv",
                index=False,
            )
            n_saved += 1

        except Exception as exc:
            n_failed += 1
            print(
                f"[SKIP] year={test_year} offset={ball_offset} "
                f"sample={sample_index} source={source_file} reason={exc!r}"
            )

    print(
        f"Completed {condition_name}: "
        f"saved={n_saved}, failed={n_failed}"
    )


# =============================================================================
# Multi-year sensitivity analysis
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset/preceding"
MODEL_ROOT = PROJECT_ROOT / "base_model_evaluation/expanding_window/models"
RAW_DATA_BASE = PROJECT_ROOT / "data/pitch_sequence"
PITCH_TYPE_BASE = PROJECT_ROOT / "data/pitch_type"

RESULT_ROOT = PROJECT_ROOT / "Sensitivity_analysis/sim2_sim"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


YEAR_CONFIGS = [
    {"test_year": 2022, "train_tag": "2018_19", "model_name": "train_2018_19_test_2022"},
    {"test_year": 2023, "train_tag": "2018_22", "model_name": "train_2018_22_test_2023"},
    {"test_year": 2024, "train_tag": "2018_23", "model_name": "train_2018_23_test_2024"},
    {"test_year": 2025, "train_tag": "2018_24", "model_name": "train_2018_24_test_2025"},
]


SIMULATION_OFFSETS = [1, 2, 3, 4, 5]


def apply_saved_platt_scaling(raw_probabilities):
    raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64).ravel()
    clipped = np.clip(raw_probabilities, probability_clip_eps, 1.0 - probability_clip_eps)
    raw_logits = np.log(clipped / (1.0 - clipped))
    calibrated_logits = platt_coef * raw_logits + platt_intercept
    calibrated_logits = np.clip(calibrated_logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def load_model_bundle(model_name):
    global model, x_mean, x_std, c_mean, c_std
    global platt_coef, platt_intercept, probability_clip_eps

    model_dir = MODEL_ROOT / model_name
    model_path = model_dir / "model.keras"
    model_info_path = model_dir / "model_info.npz"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not model_info_path.is_file():
        raise FileNotFoundError(f"Model info not found: {model_info_path}")

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "PaddingMaskLayer": PaddingMaskLayer,
            "PositionalEmbedding": PositionalEmbedding,
            "TransformerEncoderBlock": TransformerEncoderBlock,
            "MaskedMeanPooling": MaskedMeanPooling,
            "PitchSequence>PaddingMaskLayer": PaddingMaskLayer,
            "PitchSequence>PositionalEmbedding": PositionalEmbedding,
            "PitchSequence>TransformerEncoderBlock": TransformerEncoderBlock,
            "PitchSequence>MaskedMeanPooling": MaskedMeanPooling,
        },
        compile=False,
    )

    with np.load(model_info_path, allow_pickle=True) as info:
        x_mean = info["x_mean"].astype(np.float32)
        x_std = info["x_std"].astype(np.float32)
        c_mean = info["c_mean"].astype(np.float32)
        c_std = info["c_std"].astype(np.float32)

        trained_feature_columns = [
            str(value) for value in info["pitch_feature_columns"].tolist()
        ]
        if trained_feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                "Model training feature schema mismatch.\\n"
                f"Expected: {FEATURE_COLUMNS}\\n"
                f"Model   : {trained_feature_columns}"
            )

        platt_coef = float(np.asarray(info["platt_coef"]).ravel()[0])
        platt_intercept = float(np.asarray(info["platt_intercept"]).ravel()[0])
        probability_clip_eps = float(
            np.asarray(info["probability_clip_eps"]).ravel()[0]
        )

    return model_path, model_info_path


for config in YEAR_CONFIGS:
    test_year = int(config["test_year"])
    TRAIN_YEAR_TAG = config["train_tag"]
    model_name = config["model_name"]


    RAW_DATA_ROOT = RAW_DATA_BASE / str(test_year)
    PITCH_TYPE_ROOT = PITCH_TYPE_BASE / str(test_year)

    year_result_dir = RESULT_ROOT / str(test_year)
    year_result_dir.mkdir(parents=True, exist_ok=True)


    if not RAW_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_ROOT}")
    if not PITCH_TYPE_ROOT.is_dir():
        raise FileNotFoundError(f"Pitch-type directory not found: {PITCH_TYPE_ROOT}")

    model_path, model_info_path = load_model_bundle(model_name)

    # lru_cache keys contain only source_file, so caches must be cleared
    # whenever the year-specific source directories change.
    if hasattr(load_pitch_type_table, "cache_clear"):
        load_pitch_type_table.cache_clear()
    if hasattr(load_raw_source_file, "cache_clear"):
        load_raw_source_file.cache_clear()

    print("\n" + "=" * 100)
    print(f"Simulation condition: sim2_sim")
    print(f"Test year           : {test_year}")
    print(f"Model               : {model_path}")
    print(f"Model info          : {model_info_path}")

    for ball_offset in SIMULATION_OFFSETS:
        run_condition(
            ball_offset=ball_offset,
            year_result_dir=year_result_dir,
        )

print("\nFinish sim2_sim for all test years")
