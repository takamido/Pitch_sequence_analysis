# Please edit the path according to your PC environment
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models

# Define Transformer model
FEATURE_COLUMNS = [
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z_norm"]

@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class PaddingMaskLayer(layers.Layer):
    def call(self, inputs):
        is_padding = tf.reduce_all(tf.equal(inputs, 0.0), axis=-1)
        valid_mask = tf.logical_not(is_padding)
        return valid_mask

@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(
            input_dim=seq_len,
            output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos = self.pos_emb(positions)
        return x + pos

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model})
        return config

@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self, d_model,
        num_heads, ff_dim,
        dropout_rate=0.2,
        **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization()

        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)])

        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x, valid_mask = inputs
        attn_mask = tf.cast(valid_mask[:, tf.newaxis, :], tf.bool)

        attn_output = self.attn(
            x,x,
            attention_mask=attn_mask,
            training=training)
        attn_output = self.dropout1(attn_output, training=training)

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        x = self.norm2(x + ffn_output)

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

        mask = tf.cast(valid_mask, tf.float32)
        mask = mask[:, :, tf.newaxis]

        x = x * mask

        sum_x = tf.reduce_sum(x, axis=1)
        count = tf.reduce_sum(mask, axis=1)

        return sum_x / tf.maximum(count, 1.0)

#Define helper functions
def transform_sequence_standardizer(X, mean, std):
    X_scaled = X.copy().astype(np.float32)
    flat = X_scaled.reshape(-1, X_scaled.shape[-1])
    non_padding_mask = ~(np.all(flat == 0, axis=1))
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
    df["plate_z_norm"] = ((df["plate_z"] - df["sz_bot"]) / (df["sz_top"] - df["sz_bot"]))
    return df

def normalize_x_for_batter_view(df):
    df = df.copy()
    x_cols = ["pfx_x","plate_x"]
    left_mask = df["stand"] == "L"
    for col in x_cols:
        if col in df.columns:
            df.loc[left_mask, col] = -df.loc[left_mask, col]
    return df

def load_pitch_type_rank_table(source_file, year):
    path = PITCH_TYPE_RANK_ROOT / str(int(year)) / Path(source_file).name
    if not path.is_file():
        return None, f"rank file not found: {path}"

    df = pd.read_csv(path)
    required_columns = [
        "pitch_type",
        "grid_x_index", "grid_z_index",
        "effective_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x", "pfx_z",
        "plate_x", "plate_z_norm"]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in representative-pitch file {path}: {missing}")

    numeric_columns = [
        "grid_x_index", "grid_z_index",
        "effective_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x", "pfx_z",
        "plate_x", "plate_z_norm",
        "plate_x_raw", "plate_z",
        "sz_bot", "sz_top",
        "sample_count", "rank_center_score",
        "source_row_index", "game_pk",
        "at_bat_number", "pitch_number", "batter"]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()
    if df.empty:
        return None, f"no qualified representative pitches: {path}"

    df["grid_x_index"] = df["grid_x_index"].astype(int)
    df["grid_z_index"] = df["grid_z_index"].astype(int)
    return df, None

def load_raw_source_file(source_file, year):
    path = RAW_DATA_ROOT / str(int(year)) / Path(source_file).name
    df = pd.read_csv(path)

    numeric_cols = [
        "game_pk", "at_bat_number",
        "pitcher", "batter",
        "pitch_number", "strikes",
        "plate_x", "plate_z",
        "sz_top", "sz_bot",
        "effective_speed",
        "release_spin_rate",
        "spin_axis", "pfx_x", "pfx_z"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_plate_z_norm(df)
    df = normalize_x_for_batter_view(df)
    return df

def find_original_last_pitch(meta_row):
    source_file = meta_row["source_file"]
    raw_df = load_raw_source_file(source_file, meta_row["year"])
    cond = (
        (raw_df["game_pk"] == meta_row["game_pk"]) &
        (raw_df["at_bat_number"] == meta_row["at_bat_number"]) &
        (raw_df["pitcher"] == meta_row["pitcher"]) &
        (raw_df["batter"] == meta_row["batter"]))
    g = raw_df.loc[cond].copy()
    g = g.sort_values("pitch_number")
    return g.iloc[-1]

def find_previous_pitch(meta_row):
    source_file = meta_row["source_file"]
    raw_df = load_raw_source_file(source_file, meta_row["year"])
    cond = (
        (raw_df["game_pk"] == meta_row["game_pk"]) &
        (raw_df["at_bat_number"] == meta_row["at_bat_number"]) &
        (raw_df["pitcher"] == meta_row["pitcher"]) &
        (raw_df["batter"] == meta_row["batter"]))
    g = raw_df.loc[cond].copy()
    g = g.sort_values("pitch_number")
    return g.iloc[-2]

def make_simulation_rows(i, meta_row, pitch_rank_df):
    original_seq = X_test[i].copy()
    original_context = C_test[i].copy().astype(np.float32)

    original_last = find_original_last_pitch(meta_row)

    sz_top = original_last.get("sz_top", np.nan)
    sz_bot = original_last.get("sz_bot", np.nan)

    rows = []
    original_output = predict_model_output(
        original_seq.reshape(1, MAX_PITCHES, len(FEATURE_COLUMNS)),
        original_context.reshape(1, -1))[0]

    original_feature = dict(zip(FEATURE_COLUMNS, original_seq[-1]))

    rows.append({
        "row_type": "original",
        "pitch_type": original_last.get("pitch_type", "original"),
        "grid_x_index": np.nan,
        "grid_z_index": np.nan,
        "effective_speed": original_feature["effective_speed"],
        "release_spin_rate": original_feature["release_spin_rate"],
        "spin_axis": original_feature["spin_axis"],
        "pfx_x": original_feature["pfx_x"],
        "pfx_z": original_feature["pfx_z"],
        "plate_x": original_feature["plate_x"],
        "plate_z_norm": original_feature["plate_z_norm"],
        "plate_z": original_last.get("plate_z", np.nan),
        "sz_bot": sz_bot,
        "sz_top": sz_top,
        "model_output": original_output,
    })

    previous_pitch = find_previous_pitch(meta_row)
    previous_feature = dict(zip(FEATURE_COLUMNS, original_seq[-2]))

    rows.append({
        "row_type": "previous",
        "pitch_type": previous_pitch.get("pitch_type", np.nan),
        "grid_x_index": np.nan,
        "grid_z_index": np.nan,
        "effective_speed": previous_feature["effective_speed"],
        "release_spin_rate": previous_feature["release_spin_rate"],
        "spin_axis": previous_feature["spin_axis"],
        "pfx_x": previous_feature["pfx_x"],
        "pfx_z": previous_feature["pfx_z"],
        "plate_x": previous_feature["plate_x"],
        "plate_z_norm": previous_feature["plate_z_norm"],
        "plate_z": previous_pitch.get("plate_z", np.nan),
        "sz_bot": previous_pitch.get("sz_bot", sz_bot),
        "sz_top": previous_pitch.get("sz_top", sz_top),
        "model_output": np.nan,
    })

    # No artificial 3x3 placement is performed here.
    # Each row in pitch_rank_df is one actually observed representative pitch
    # for a pitcher x pitch_type x qualified grid cell.
    if pitch_rank_df is None or pitch_rank_df.empty:
        return pd.DataFrame(rows), "no representative-pitch candidates"

    candidate_seqs = []
    candidate_meta = []

    # The rank-extraction file stores pfx_x in the original Statcast sign,
    # while model inputs use batter-view normalization for pfx_x.
    # Convert pfx_x for the batter of the simulated plate appearance.
    stand = original_last.get("stand", None)
    x_sign = -1.0 if stand == "L" else 1.0

    for _, pt in pitch_rank_df.iterrows():
        seq_cf = original_seq.copy()

        model_pfx_x = float(pt["pfx_x"]) * x_sign
        plate_x = float(pt["plate_x"])
        plate_z_norm = float(pt["plate_z_norm"])

        seq_cf[-1, FEATURE_COLUMNS.index("effective_speed")] = pt["effective_speed"]
        seq_cf[-1, FEATURE_COLUMNS.index("release_spin_rate")] = pt["release_spin_rate"]
        seq_cf[-1, FEATURE_COLUMNS.index("spin_axis")] = pt["spin_axis"]
        seq_cf[-1, FEATURE_COLUMNS.index("pfx_x")] = model_pfx_x
        seq_cf[-1, FEATURE_COLUMNS.index("pfx_z")] = pt["pfx_z"]
        seq_cf[-1, FEATURE_COLUMNS.index("plate_x")] = plate_x
        seq_cf[-1, FEATURE_COLUMNS.index("plate_z_norm")] = plate_z_norm

        # Convert the representative pitch's normalized vertical location to
        # the strike-zone height of the current simulated batter.
        plate_z_actual = sz_bot + plate_z_norm * (sz_top - sz_bot)

        candidate_seqs.append(seq_cf)
        candidate_meta.append({
            "row_type": "simulation",
            "pitch_type": pt["pitch_type"],
            "grid_x_index": int(pt["grid_x_index"]),
            "grid_z_index": int(pt["grid_z_index"]),
            "effective_speed": pt["effective_speed"],
            "release_spin_rate": pt["release_spin_rate"],
            "spin_axis": pt["spin_axis"],
            "pfx_x": model_pfx_x,
            "pfx_z": pt["pfx_z"],
            "plate_x": plate_x,
            "plate_z_norm": plate_z_norm,
            "plate_z": plate_z_actual,
            "sz_bot": sz_bot,
            "sz_top": sz_top,
            # Traceability back to the actually observed representative pitch.
            "representative_stand": pt.get("stand", np.nan),
            "representative_plate_x_raw": pt.get("plate_x_raw", np.nan),
            "representative_plate_z_raw": pt.get("plate_z", np.nan),
            "representative_sample_count": pt.get("sample_count", np.nan),
            "representative_rank_center_score": pt.get("rank_center_score", np.nan),
            "representative_source_row_index": pt.get("source_row_index", np.nan),
            "representative_game_date": pt.get("game_date", np.nan),
            "representative_game_pk": pt.get("game_pk", np.nan),
            "representative_at_bat_number": pt.get("at_bat_number", np.nan),
            "representative_pitch_number": pt.get("pitch_number", np.nan),
            "representative_batter": pt.get("batter", np.nan),
        })

    candidate_seqs = np.stack(candidate_seqs).astype(np.float32)
    candidate_contexts = np.repeat(
        original_context.reshape(1, -1),
        len(candidate_seqs),
        axis=0).astype(np.float32)

    outputs = predict_model_output(candidate_seqs, candidate_contexts)

    for candidate_row, output in zip(candidate_meta, outputs):
        candidate_row["model_output"] = output
        rows.append(candidate_row)

    result_df = pd.DataFrame(rows)
    return result_df, None


# =============================================================================
# Multi-year sensitivity analysis: simulation 1 / observed representative pitch
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset/last"
MODEL_ROOT = PROJECT_ROOT / "base_model_evaluation/expanding_window/models"
RAW_DATA_ROOT = PROJECT_ROOT / "data/pitch_sequence"
PITCH_TYPE_RANK_ROOT = PROJECT_ROOT / "data/pitch_type_act_inzone"

RESULT_ROOT = PROJECT_ROOT / "Sensitivity_analysis/sim1_act"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


YEAR_CONFIGS = [
    {"test_year": 2022, "train_tag": "2018_19", "model_name": "train_2018_19_test_2022"},
    {"test_year": 2023, "train_tag": "2018_22", "model_name": "train_2018_22_test_2023"},
    {"test_year": 2024, "train_tag": "2018_23", "model_name": "train_2018_23_test_2024"},
    {"test_year": 2025, "train_tag": "2018_24", "model_name": "train_2018_24_test_2025"},
]


MAX_PITCHES = 6


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
    train_tag = config["train_tag"]
    model_name = config["model_name"]

    DATASET_PATH = DATASET_ROOT / f"dataset_last_{train_tag}.npz"
    META_TEST_PATH = DATASET_ROOT / f"meta_test_last_{train_tag}.csv"
    year_result_dir = RESULT_ROOT / str(test_year)
    year_result_dir.mkdir(parents=True, exist_ok=True)

    for required_path, label in [
        (DATASET_PATH, "dataset"),
        (META_TEST_PATH, "test metadata"),
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(f"{label} file not found: {required_path}")

    for required_dir, label in [
        (RAW_DATA_ROOT / str(test_year), "raw pitch data"),
        (PITCH_TYPE_RANK_ROOT / str(test_year), "representative-pitch data"),
    ]:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {required_dir}")

    model_path, model_info_path = load_model_bundle(model_name)

    with np.load(DATASET_PATH, allow_pickle=True) as data:
        X_test = data["X_test"].astype(np.float32)
        C_test = data["C_test"].astype(np.float32)

    meta_test = pd.read_csv(META_TEST_PATH)
    if len(meta_test) != len(X_test):
        raise ValueError(
            f"Length mismatch for {test_year}: meta={len(meta_test)}, X_test={len(X_test)}"
        )

    print("\n" + "=" * 100)
    print(f"Simulation condition: sim1_act")
    print(f"Test year           : {test_year}")
    print(f"Model               : {model_path}")
    print(f"Model info          : {model_info_path}")
    print(f"Samples             : {len(meta_test)}")

    for sample_index, meta_row in tqdm(
        meta_test.iterrows(),
        total=len(meta_test),
        desc=f"sim1_act {test_year}",
    ):
        source_file = Path(str(meta_row["source_file"])).name
        try:
            pitch_rank_df, load_reason = load_pitch_type_rank_table(
                source_file, test_year
            )
            result_df, simulation_reason = make_simulation_rows(
                sample_index, meta_row, pitch_rank_df
            )
            reason = load_reason or simulation_reason
            if reason is not None:
                raise ValueError(reason)

            result_df.to_csv(
                year_result_dir / f"{sample_index:06d}.csv",
                index=False,
            )
        except Exception as exc:
            print(
                f"[SKIP] year={test_year} sample={sample_index} "
                f"source={source_file} reason={exc!r}"
            )

print("\nFinish sim1_act for all test years")
