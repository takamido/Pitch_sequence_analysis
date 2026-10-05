import gc
import os
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


# =============================================================================
# Paths and settings
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

DATASET_PATH = Path("./dataset/last/dataset_last_2018_24.npz")
META_TRAIN_PATH = Path("./dataset/last/meta_train_last_2018_24.csv")
META_TEST_PATH = Path("./dataset/last/meta_test_last_2018_24.csv")
EXPANDING_DATA_DIR = Path("./dataset/last")
RAW_PITCH_DATA_ROOT = Path("./data/pitch_sequence")
PITCHER_STATS_PATH = Path("./data/stats/pitcher/2025.xlsx")

OUTPUT_ROOT = Path("./base_model_evaluation")
FULL_MODEL_DIR = OUTPUT_ROOT / "full_model"
EXPANDING_WINDOW_DIR = OUTPUT_ROOT / "expanding_window"
EXPANDING_MODELS_DIR = EXPANDING_WINDOW_DIR / "models"
SUBGROUP_DIR = OUTPUT_ROOT / "subgroup_analysis"

CONTACT_THRESHOLD = 0.0
CLASSIFICATION_THRESHOLD = 0.5
VAL_SIZE = 0.1
MAX_PITCHES = 6
RANDOM_SEED = 0

CALIBRATION_N_BINS = 10
BOOTSTRAP_N = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

CONTEXT_DENSE_UNITS = 32
DROPOUT_RATE = 0.0
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
EPOCHS = 200
PATIENCE = 10

MIN_PITCH_TYPE_SAMPLES = 200
STRIKE_ZONE_HALF_WIDTH_FT = 0.83
EXPECTED_MIN_PITCHER_SAMPLE_COVERAGE = 0.95
PITCHER_K_GROUP_ORDER = ["Lower third", "Middle third", "Upper third"]

HYPERPARAMETER_CANDIDATES = [
    {"d_model": 64, "num_heads": 2, "ff_dim": 128, "num_blocks": 1, "dense_units": 64},
    {"d_model": 64, "num_heads": 4, "ff_dim": 128, "num_blocks": 1, "dense_units": 64},
    {"d_model": 64, "num_heads": 4, "ff_dim": 256, "num_blocks": 2, "dense_units": 64},
    {"d_model": 128, "num_heads": 4, "ff_dim": 256, "num_blocks": 2, "dense_units": 64},
    {"d_model": 128, "num_heads": 4, "ff_dim": 512, "num_blocks": 2, "dense_units": 64},
    {"d_model": 256, "num_heads": 4, "ff_dim": 512, "num_blocks": 2, "dense_units": 64},
    {"d_model": 256, "num_heads": 8, "ff_dim": 512, "num_blocks": 2, "dense_units": 64},
    {"d_model": 256, "num_heads": 8, "ff_dim": 1024, "num_blocks": 2, "dense_units": 128},
]

EXPANDING_WINDOW_CONFIGS = [
    {"train_tag": "2018_19", "train_years": [2018, 2019], "test_year": 2022},
    {"train_tag": "2018_22", "train_years": [2018, 2019, 2022], "test_year": 2023},
    {"train_tag": "2018_23", "train_years": [2018, 2019, 2022, 2023], "test_year": 2024},
    {"train_tag": "2018_24", "train_years": [2018, 2019, 2022, 2023, 2024], "test_year": 2025},
]

PITCH_TYPE_NAME_MAP = {
    "FF": "Four-Seam Fastball", "FA": "Fastball", "SI": "Sinker",
    "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "KC": "Knuckle Curve", "CS": "Slow Curve", "CH": "Changeup",
    "FS": "Split-Finger", "FO": "Forkball", "SV": "Slurve",
    "KN": "Knuckleball", "EP": "Eephus", "SC": "Screwball",
}
HAND_NAME_MAP = {
    "L": "Left-handed batter",
    "R": "Right-handed batter",
    "S": "Switch hitter",
}

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


# =============================================================================
# Reproducibility and preprocessing
# =============================================================================
def set_global_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def fit_sequence_standardizer(X):
    X_flat = X.reshape(-1, X.shape[-1])
    non_padding_mask = ~np.all(X_flat == 0, axis=1)
    X_valid = X_flat[non_padding_mask]
    mean = X_valid.mean(axis=0)
    std = X_valid.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_sequence_standardizer(X, mean, std):
    X_scaled = X.copy().astype(np.float32)
    flat = X_scaled.reshape(-1, X_scaled.shape[-1])
    non_padding_mask = ~np.all(flat == 0, axis=1)
    flat[non_padding_mask] = (flat[non_padding_mask] - mean) / std
    return X_scaled


def fit_tabular_standardizer(C):
    mean = C.mean(axis=0)
    std = C.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_tabular_standardizer(C, mean, std):
    return ((C.astype(np.float32) - mean) / std).astype(np.float32)


def valid_pitch_mask(sequence):
    return ~np.all(sequence == 0, axis=1)


def keep_last_n_pitches(X, n_pitches):
    transformed = np.zeros_like(X)
    for sample_idx, sequence in enumerate(X):
        valid = sequence[valid_pitch_mask(sequence)]
        kept = valid[-n_pitches:]
        transformed[sample_idx, -len(kept):] = kept
    return transformed


# =============================================================================
# Transformer model
# =============================================================================
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
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


@tf.keras.utils.register_keras_serializable(package="PitchSequence")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.0, **kwargs):
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
        attn_output = self.attn(x, x, attention_mask=attn_mask, training=training)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x, training=training)
        return self.norm2(x + self.dropout2(ffn_output, training=training))

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


def build_transformer_classifier_with_context(
    pitch_input_shape,
    context_input_shape,
    d_model,
    num_heads,
    ff_dim,
    num_blocks,
    dense_units,
    context_dense_units=CONTEXT_DENSE_UNITS,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE,
):
    pitch_inputs = layers.Input(shape=pitch_input_shape, name="pitch_sequence")
    valid_mask = PaddingMaskLayer(name="padding_mask")(pitch_inputs)

    x = layers.Dense(d_model, name="pitch_projection")(pitch_inputs)
    x = PositionalEmbedding(
        seq_len=pitch_input_shape[0], d_model=d_model,
        name="positional_embedding",
    )(x)

    for block_idx in range(num_blocks):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_encoder_block_{block_idx + 1}",
        )([x, valid_mask])

    pitch_repr = MaskedMeanPooling(name="masked_mean_pooling")([x, valid_mask])
    pitch_repr = layers.Dense(dense_units, activation="relu", name="pitch_repr_dense")(pitch_repr)
    pitch_repr = layers.Dropout(dropout_rate, name="pitch_repr_dropout")(pitch_repr)

    context_inputs = layers.Input(shape=context_input_shape, name="context_features")
    context_repr = layers.Dense(
        context_dense_units, activation="relu", name="context_repr_dense"
    )(context_inputs)
    context_repr = layers.Dropout(dropout_rate, name="context_repr_dropout")(context_repr)

    merged = layers.Concatenate(name="pitch_context_concat")([pitch_repr, context_repr])
    z = layers.Dense(dense_units, activation="relu", name="merged_dense")(merged)
    z = layers.Dropout(dropout_rate, name="merged_dropout")(z)
    outputs = layers.Dense(1, activation="sigmoid", name="in_play_probability")(z)

    model = models.Model(
        inputs=[pitch_inputs, context_inputs],
        outputs=outputs,
        name="transformer_binary_classifier_with_context_C",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# =============================================================================
# Calibration and bootstrap evaluation
# =============================================================================
def probability_to_logit(y_prob, eps=1e-6):
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return np.log(y_prob / (1.0 - y_prob))


def fit_platt_scaler(y_val_true, y_val_prob_raw):
    y_val_true = np.asarray(y_val_true).astype(int).ravel()
    val_logits = probability_to_logit(y_val_prob_raw).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(val_logits, y_val_true)
    return calibrator


def apply_platt_scaler(calibrator, y_prob_raw):
    logits = probability_to_logit(y_prob_raw).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1].astype(np.float64)


def calculate_calibration_statistics(y_true, y_prob, n_bins=CALIBRATION_N_BINS):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    brier_score = float(np.mean((y_prob - y_true) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)

    rows = []
    ece = 0.0
    n_samples = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        lower = float(bin_edges[bin_idx])
        upper = float(bin_edges[bin_idx + 1])

        if count > 0:
            mean_pred = float(y_prob[mask].mean())
            observed_rate = float(y_true[mask].mean())
            absolute_gap = abs(observed_rate - mean_pred)
            sample_fraction = count / n_samples
            ece += sample_fraction * absolute_gap
        else:
            mean_pred = np.nan
            observed_rate = np.nan
            absolute_gap = np.nan
            sample_fraction = 0.0

        rows.append({
            "bin_index": bin_idx + 1,
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "sample_fraction": float(sample_fraction),
            "mean_predicted_probability": mean_pred,
            "observed_positive_rate": observed_rate,
            "absolute_calibration_gap": absolute_gap,
        })

    return brier_score, float(ece), pd.DataFrame(rows)


def calculate_metrics(y_true, y_prob, threshold=CLASSIFICATION_THRESHOLD):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc = (
        float(roc_auc_score(y_true, y_prob))
        if np.unique(y_true).size >= 2
        else np.nan
    )
    brier_score, ece, _ = calculate_calibration_statistics(y_true, y_prob)

    return {
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(y_prob.mean()),
        "calibration_gap": float(y_prob.mean() - y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "brier_score": float(brier_score),
        "ece": float(ece),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def generate_bootstrap_seeds(n_bootstrap, base_seed):
    child_sequences = np.random.SeedSequence(int(base_seed)).spawn(int(n_bootstrap))
    return np.asarray([
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in child_sequences
    ], dtype=np.uint64)


def bootstrap_evaluate(y_true, y_prob, threshold=CLASSIFICATION_THRESHOLD):
    """Return one compact metrics row and calibration-bin table with 95% CIs.

    Bootstrap replicate-level data are kept only in memory and are not written to disk.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    point_metrics = calculate_metrics(y_true, y_prob, threshold)
    _, _, point_calibration = calculate_calibration_statistics(y_true, y_prob)

    metric_names = [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "brier_score", "ece",
    ]
    metric_samples = {name: [] for name in metric_names}
    calibration_samples = {
        bin_idx: {
            "count": [],
            "sample_fraction": [],
            "mean_predicted_probability": [],
            "observed_positive_rate": [],
            "absolute_calibration_gap": [],
        }
        for bin_idx in range(1, CALIBRATION_N_BINS + 1)
    }

    sample_size = len(y_true)
    bootstrap_seeds = generate_bootstrap_seeds(BOOTSTRAP_N, RANDOM_SEED)

    for bootstrap_seed in bootstrap_seeds:
        rng = np.random.default_rng(int(bootstrap_seed))
        sampled_indices = rng.integers(0, sample_size, size=sample_size, endpoint=False)
        sampled_y_true = y_true[sampled_indices]
        sampled_y_prob = y_prob[sampled_indices]

        replicate_metrics = calculate_metrics(sampled_y_true, sampled_y_prob, threshold)
        _, _, replicate_calibration = calculate_calibration_statistics(
            sampled_y_true, sampled_y_prob
        )

        for metric_name in metric_names:
            metric_samples[metric_name].append(replicate_metrics[metric_name])

        for row in replicate_calibration.itertuples(index=False):
            bucket = calibration_samples[int(row.bin_index)]
            for column in bucket:
                bucket[column].append(getattr(row, column))

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    metrics_row = dict(point_metrics)
    metrics_row["confidence_level"] = BOOTSTRAP_CONFIDENCE_LEVEL
    metrics_row["bootstrap_n"] = BOOTSTRAP_N

    for metric_name in metric_names:
        values = np.asarray(metric_samples[metric_name], dtype=float)
        values = values[np.isfinite(values)]
        metrics_row[f"{metric_name}_ci_lower"] = (
            float(np.quantile(values, alpha)) if len(values) else np.nan
        )
        metrics_row[f"{metric_name}_ci_upper"] = (
            float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
        )

    calibration_rows = []
    for point_row in point_calibration.itertuples(index=False):
        bin_idx = int(point_row.bin_index)
        result = {
            "bin_index": bin_idx,
            "bin_lower": float(point_row.bin_lower),
            "bin_upper": float(point_row.bin_upper),
            "count": int(point_row.count),
            "sample_fraction": float(point_row.sample_fraction),
            "mean_predicted_probability": point_row.mean_predicted_probability,
            "observed_positive_rate": point_row.observed_positive_rate,
            "absolute_calibration_gap": point_row.absolute_calibration_gap,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "bootstrap_n": BOOTSTRAP_N,
        }

        for column in [
            "count", "sample_fraction", "mean_predicted_probability",
            "observed_positive_rate", "absolute_calibration_gap",
        ]:
            values = np.asarray(calibration_samples[bin_idx][column], dtype=float)
            values = values[np.isfinite(values)]
            result[f"{column}_ci_lower"] = (
                float(np.quantile(values, alpha)) if len(values) else np.nan
            )
            result[f"{column}_ci_upper"] = (
                float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
            )

        calibration_rows.append(result)

    return metrics_row, pd.DataFrame(calibration_rows)


def evaluate_raw_and_platt(y_true, y_prob_raw, y_prob_platt, metadata=None):
    metadata = {} if metadata is None else dict(metadata)
    metric_rows = []
    calibration_tables = []

    for calibration_method, probabilities in [
        ("raw", y_prob_raw),
        ("platt", y_prob_platt),
    ]:
        metrics_row, calibration_df = bootstrap_evaluate(y_true, probabilities)
        metrics_row = {
            **metadata,
            "calibration_method": calibration_method,
            "platt_applied": calibration_method == "platt",
            **metrics_row,
        }
        metric_rows.append(metrics_row)

        calibration_df.insert(0, "calibration_method", calibration_method)
        calibration_df.insert(1, "platt_applied", calibration_method == "platt")
        for column_name, column_value in reversed(list(metadata.items())):
            calibration_df.insert(0, column_name, column_value)
        calibration_tables.append(calibration_df)

    return pd.DataFrame(metric_rows), pd.concat(calibration_tables, ignore_index=True)


# =============================================================================
# Hyperparameter tuning and model fitting
# =============================================================================
def make_model_inputs(X_scaled, C_scaled):
    return {
        "pitch_sequence": X_scaled,
        "context_features": C_scaled,
    }


def run_hyperparameter_search(X_base, C_base, y_base, train_indices, val_indices):
    X_tr = X_base[train_indices]
    X_val = X_base[val_indices]
    C_tr = C_base[train_indices]
    C_val = C_base[val_indices]
    y_tr = y_base[train_indices]
    y_val = y_base[val_indices]

    x_mean, x_std = fit_sequence_standardizer(X_tr)
    c_mean, c_std = fit_tabular_standardizer(C_tr)
    X_tr_scaled = transform_sequence_standardizer(X_tr, x_mean, x_std)
    X_val_scaled = transform_sequence_standardizer(X_val, x_mean, x_std)
    C_tr_scaled = transform_tabular_standardizer(C_tr, c_mean, c_std)
    C_val_scaled = transform_tabular_standardizer(C_val, c_mean, c_std)

    search_results = []
    for trial_id, params in enumerate(HYPERPARAMETER_CANDIDATES, start=1):
        print(f"Hyperparameter trial {trial_id}/{len(HYPERPARAMETER_CANDIDATES)}: {params}")
        tf.keras.backend.clear_session()
        set_global_seed(RANDOM_SEED + trial_id)

        model = build_transformer_classifier_with_context(
            pitch_input_shape=X_tr_scaled.shape[1:],
            context_input_shape=C_tr_scaled.shape[1:],
            **params,
        )
        early_stop = callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
        )
        history = model.fit(
            make_model_inputs(X_tr_scaled, C_tr_scaled),
            y_tr,
            validation_data=(make_model_inputs(X_val_scaled, C_val_scaled), y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )

        best_epoch_idx = int(np.argmax(history.history["val_auc"]))
        search_results.append({
            "trial_id": trial_id,
            **params,
            "best_val_auc": float(history.history["val_auc"][best_epoch_idx]),
            "best_val_loss": float(history.history["val_loss"][best_epoch_idx]),
        })
        del model
        gc.collect()

    search_df = pd.DataFrame(search_results).sort_values(
        ["best_val_auc", "best_val_loss"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = search_df.iloc[0]
    return {
        "d_model": int(best["d_model"]),
        "num_heads": int(best["num_heads"]),
        "ff_dim": int(best["ff_dim"]),
        "num_blocks": int(best["num_blocks"]),
        "dense_units": int(best["dense_units"]),
    }


def save_model_bundle(
    model,
    output_dir,
    x_mean,
    x_std,
    c_mean,
    c_std,
    platt_scaler,
    selected_params,
    pitch_feature_columns,
    c_feature_columns,
):
    """Save only what is needed to reuse an expanding-window model later."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "model.keras")
    np.savez_compressed(
        output_dir / "model_info.npz",
        x_mean=x_mean,
        x_std=x_std,
        c_mean=c_mean,
        c_std=c_std,
        pitch_feature_columns=np.asarray(pitch_feature_columns),
        c_feature_columns=np.asarray(c_feature_columns),
        classification_threshold=np.array(CLASSIFICATION_THRESHOLD, dtype=np.float32),
        contact_threshold=np.array(CONTACT_THRESHOLD, dtype=np.float32),
        max_pitches=np.array(MAX_PITCHES, dtype=np.int32),
        d_model=np.array(selected_params["d_model"], dtype=np.int32),
        num_heads=np.array(selected_params["num_heads"], dtype=np.int32),
        ff_dim=np.array(selected_params["ff_dim"], dtype=np.int32),
        num_blocks=np.array(selected_params["num_blocks"], dtype=np.int32),
        dense_units=np.array(selected_params["dense_units"], dtype=np.int32),
        context_dense_units=np.array(CONTEXT_DENSE_UNITS, dtype=np.int32),
        platt_coef=platt_scaler.coef_.astype(np.float64),
        platt_intercept=platt_scaler.intercept_.astype(np.float64),
        platt_classes=platt_scaler.classes_.astype(int),
        platt_score_transform=np.array("logit(raw_probability)"),
        probability_clip_eps=np.array(1e-6, dtype=np.float64),
    )


def train_evaluate_model(
    X_train,
    C_train,
    y_train_cls,
    X_test,
    C_test,
    y_test_cls,
    meta_test,
    train_indices,
    val_indices,
    selected_params,
    pitch_feature_columns,
    c_feature_columns,
    metadata,
    model_save_dir=None,
):
    X_tr = X_train[train_indices]
    X_val = X_train[val_indices]
    C_tr = C_train[train_indices]
    C_val = C_train[val_indices]
    y_tr = y_train_cls[train_indices]
    y_val = y_train_cls[val_indices]

    x_mean, x_std = fit_sequence_standardizer(X_tr)
    c_mean, c_std = fit_tabular_standardizer(C_tr)

    X_tr_scaled = transform_sequence_standardizer(X_tr, x_mean, x_std)
    X_val_scaled = transform_sequence_standardizer(X_val, x_mean, x_std)
    X_test_scaled = transform_sequence_standardizer(X_test, x_mean, x_std)
    C_tr_scaled = transform_tabular_standardizer(C_tr, c_mean, c_std)
    C_val_scaled = transform_tabular_standardizer(C_val, c_mean, c_std)
    C_test_scaled = transform_tabular_standardizer(C_test, c_mean, c_std)

    tf.keras.backend.clear_session()
    set_global_seed(RANDOM_SEED)
    model = build_transformer_classifier_with_context(
        pitch_input_shape=X_tr_scaled.shape[1:],
        context_input_shape=C_tr_scaled.shape[1:],
        **selected_params,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    start_time = time.time()
    history = model.fit(
        make_model_inputs(X_tr_scaled, C_tr_scaled),
        y_tr,
        validation_data=(make_model_inputs(X_val_scaled, C_val_scaled), y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    elapsed_sec = time.time() - start_time

    y_val_prob_raw = model.predict(
        make_model_inputs(X_val_scaled, C_val_scaled),
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()
    platt_scaler = fit_platt_scaler(y_val, y_val_prob_raw)

    y_test_prob_raw = model.predict(
        make_model_inputs(X_test_scaled, C_test_scaled),
        batch_size=BATCH_SIZE,
        verbose=1,
    ).ravel()
    y_test_prob_platt = apply_platt_scaler(platt_scaler, y_test_prob_raw)

    common_metadata = {
        **metadata,
        "n_train_total": int(len(y_train_cls)),
        "n_train_fit": int(len(train_indices)),
        "n_validation": int(len(val_indices)),
        "n_test": int(len(y_test_cls)),
        "trained_epochs": int(len(history.history["loss"])),
        "elapsed_sec": float(elapsed_sec),
        **selected_params,
        "context_dense_units": CONTEXT_DENSE_UNITS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }

    metrics_df, calibration_df = evaluate_raw_and_platt(
        y_true=y_test_cls,
        y_prob_raw=y_test_prob_raw,
        y_prob_platt=y_test_prob_platt,
        metadata=common_metadata,
    )

    if model_save_dir is not None:
        save_model_bundle(
            model=model,
            output_dir=model_save_dir,
            x_mean=x_mean,
            x_std=x_std,
            c_mean=c_mean,
            c_std=c_std,
            platt_scaler=platt_scaler,
            selected_params=selected_params,
            pitch_feature_columns=pitch_feature_columns,
            c_feature_columns=c_feature_columns,
        )

    prediction_df = meta_test.copy()
    prediction_df["y_true_cls"] = y_test_cls.astype(int)
    prediction_df["pred_prob_raw"] = y_test_prob_raw
    prediction_df["pred_prob_platt"] = y_test_prob_platt

    del model, platt_scaler
    gc.collect()
    return metrics_df, calibration_df, prediction_df


# =============================================================================
# Subgroup helpers
# =============================================================================
def normalize_merge_keys(df, key_columns):
    df = df.copy()
    for column in key_columns:
        if column == "source_file":
            df[column] = df[column].astype(str)
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df


def resolve_unicode_filename(directory, requested_name):
    directory = Path(directory)
    direct_path = directory / str(requested_name)
    if direct_path.is_file():
        return direct_path

    target = unicodedata.normalize("NFC", str(requested_name))
    matches = [
        path for path in directory.iterdir()
        if path.is_file() and unicodedata.normalize("NFC", path.name) == target
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve {requested_name} in {directory}")
    return matches[0]


def load_last_pitch_raw_metadata(prediction_df, raw_data_root, test_year):
    merge_keys = ["source_file", "game_pk", "at_bat_number", "pitcher", "batter"]
    raw_year_dir = raw_data_root / str(test_year)
    parts = []

    for source_file in sorted(prediction_df["source_file"].dropna().astype(str).unique()):
        csv_path = resolve_unicode_filename(raw_year_dir, source_file)
        raw_df = pd.read_csv(
            csv_path,
            usecols=[*merge_keys[1:], "pitch_number", "pitch_type", "stand"],
        )
        for column in [*merge_keys[1:], "pitch_number"]:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

        last_pitch_df = (
            raw_df.sort_values([*merge_keys[1:], "pitch_number"])
            .groupby(merge_keys[1:], sort=False, dropna=False)
            .tail(1)
            .copy()
        )
        last_pitch_df.insert(0, "source_file", source_file)
        last_pitch_df = last_pitch_df.rename(columns={
            "pitch_type": "final_pitch_type_code",
            "stand": "batter_stand",
        })
        parts.append(last_pitch_df[merge_keys + ["final_pitch_type_code", "batter_stand"]])

    raw_last_pitch_df = pd.concat(parts, ignore_index=True)
    enriched_df = normalize_merge_keys(prediction_df, merge_keys).merge(
        normalize_merge_keys(raw_last_pitch_df, merge_keys),
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    assert enriched_df[["final_pitch_type_code", "batter_stand"]].notna().all().all()
    return enriched_df


def add_model_input_final_location(prediction_df, X_test, pitch_feature_columns):
    pitch_columns = [str(column) for column in pitch_feature_columns]
    plate_x_index = pitch_columns.index("plate_x")
    plate_z_norm_index = pitch_columns.index("plate_z_norm")

    enriched_df = prediction_df.copy()
    enriched_df["final_plate_x_model_input"] = X_test[:, -1, plate_x_index]
    enriched_df["final_plate_z_norm_model_input"] = X_test[:, -1, plate_z_norm_index]
    return enriched_df


def classify_strike_zone_from_model_input(row):
    horizontally_inside = abs(float(row["final_plate_x_model_input"])) <= STRIKE_ZONE_HALF_WIDTH_FT
    vertically_inside = 0.0 <= float(row["final_plate_z_norm_model_input"]) <= 1.0
    return "Inside" if horizontally_inside and vertically_inside else "Outside"


def evaluate_subgroups(
    analysis_df,
    group_column,
    ordered_groups,
    factor_name,
    extra_group_metadata=None,
):
    metric_tables = []
    calibration_tables = []
    extra_group_metadata = {} if extra_group_metadata is None else extra_group_metadata

    for group_name in ordered_groups:
        group_df = analysis_df[analysis_df[group_column] == group_name].copy()
        if group_df.empty:
            continue

        metadata = {
            "factor": factor_name,
            "group": str(group_name),
            **extra_group_metadata.get(str(group_name), {}),
        }
        metrics_df, calibration_df = evaluate_raw_and_platt(
            y_true=group_df["y_true_cls"].to_numpy(),
            y_prob_raw=group_df["pred_prob_raw"].to_numpy(),
            y_prob_platt=group_df["pred_prob_platt"].to_numpy(),
            metadata=metadata,
        )
        metric_tables.append(metrics_df)
        calibration_tables.append(calibration_df)

    if not metric_tables:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(metric_tables, ignore_index=True),
        pd.concat(calibration_tables, ignore_index=True),
    )


def load_and_prepare_pitcher_stats(stats_path, sheet_name=0):
    pitcher_stats = pd.read_excel(stats_path, sheet_name=sheet_name).copy()
    pitcher_stats["player_id"] = pd.to_numeric(pitcher_stats["player_id"]).astype("Int64")
    pitcher_stats["strikeOuts"] = pd.to_numeric(pitcher_stats["strikeOuts"])
    pitcher_stats["battersFaced"] = pd.to_numeric(pitcher_stats["battersFaced"])
    pitcher_stats = pitcher_stats.dropna(
        subset=["player_id", "strikeOuts", "battersFaced"]
    ).copy()
    pitcher_stats = pitcher_stats[pitcher_stats["battersFaced"] > 0].copy()
    pitcher_stats["pitcher_k_rate_2025"] = (
        pitcher_stats["strikeOuts"] / pitcher_stats["battersFaced"]
    )
    pitcher_stats["pitcher_k_rate_group"] = pd.qcut(
        pitcher_stats["pitcher_k_rate_2025"],
        q=3,
        labels=PITCHER_K_GROUP_ORDER,
    ).astype("string")
    return pitcher_stats[
        ["player_id", "Name", "Team", "strikeOuts", "battersFaced",
         "pitcher_k_rate_2025", "pitcher_k_rate_group"]
    ].copy()


# =============================================================================
# Main analysis
# =============================================================================
def main():
    set_global_seed(RANDOM_SEED)
    for directory in [FULL_MODEL_DIR, EXPANDING_WINDOW_DIR, EXPANDING_MODELS_DIR, SUBGROUP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load the largest training-window dataset and tune the Full model.
    # -------------------------------------------------------------------------
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        X_train = data["X_train"].astype(np.float32)
        C_train = data["C_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        C_test = data["C_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        pitch_feature_columns = data["pitch_feature_columns"].copy()
        c_feature_columns = data["c_feature_columns"].copy()

    meta_train = pd.read_csv(META_TRAIN_PATH)
    meta_test = pd.read_csv(META_TEST_PATH)
    assert X_train.shape[1] == X_test.shape[1] == MAX_PITCHES
    assert len(meta_train) == len(X_train) and len(meta_test) == len(X_test)

    y_train_cls = (y_train > CONTACT_THRESHOLD).astype(np.float32)
    y_test_cls = (y_test > CONTACT_THRESHOLD).astype(np.float32)
    all_indices = np.arange(len(y_train_cls))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_train_cls,
    )

    X_train_full = keep_last_n_pitches(X_train, MAX_PITCHES)
    X_test_full = keep_last_n_pitches(X_test, MAX_PITCHES)

    print("\n===== Hyperparameter tuning for Full model =====")
    selected_params = run_hyperparameter_search(
        X_base=X_train_full,
        C_base=C_train,
        y_base=y_train_cls,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    print("Selected parameters:", selected_params)

    # -------------------------------------------------------------------------
    # Full model. This is also the 2018-2024 -> 2025 expanding-window model,
    # so it is trained only once and saved under expanding_window/models/.
    # -------------------------------------------------------------------------
    full_model_name = "train_2018_24_test_2025"
    full_model_save_dir = EXPANDING_MODELS_DIR / full_model_name

    full_metrics_df, full_calibration_df, full_prediction_df = train_evaluate_model(
        X_train=X_train_full,
        C_train=C_train,
        y_train_cls=y_train_cls,
        X_test=X_test_full,
        C_test=C_test,
        y_test_cls=y_test_cls,
        meta_test=meta_test,
        train_indices=train_indices,
        val_indices=val_indices,
        selected_params=selected_params,
        pitch_feature_columns=pitch_feature_columns,
        c_feature_columns=c_feature_columns,
        metadata={
            "model": "full_model",
            "train_years": "2018,2019,2022,2023,2024",
            "test_year": 2025,
        },
        model_save_dir=full_model_save_dir,
    )

    full_metrics_df.to_csv(FULL_MODEL_DIR / "metrics_full_model.csv", index=False)
    full_calibration_df.to_csv(FULL_MODEL_DIR / "calibration_full_model.csv", index=False)

    # -------------------------------------------------------------------------
    # Expanding-window evaluation. Train the first three windows here and reuse
    # the already-trained Full model for the final 2025 window.
    # -------------------------------------------------------------------------
    expanding_metric_tables = []
    expanding_calibration_tables = []

    for config in EXPANDING_WINDOW_CONFIGS[:-1]:
        train_tag = config["train_tag"]
        train_years = config["train_years"]
        test_year = config["test_year"]
        model_name = f"train_{train_tag}_test_{test_year}"

        print(f"\n===== Expanding window: {train_years} -> {test_year} =====")
        dataset_path = EXPANDING_DATA_DIR / f"dataset_last_{train_tag}.npz"
        meta_test_path = EXPANDING_DATA_DIR / f"meta_test_last_{train_tag}.csv"

        with np.load(dataset_path, allow_pickle=True) as annual_data:
            X_train_year = annual_data["X_train"].astype(np.float32)
            C_train_year = annual_data["C_train"].astype(np.float32)
            y_train_year = annual_data["y_train"].astype(np.float32)
            X_test_year = annual_data["X_test"].astype(np.float32)
            C_test_year = annual_data["C_test"].astype(np.float32)
            y_test_year = annual_data["y_test"].astype(np.float32)
            annual_pitch_feature_columns = annual_data["pitch_feature_columns"].copy()
            annual_c_feature_columns = annual_data["c_feature_columns"].copy()

        meta_test_year = pd.read_csv(meta_test_path)
        y_train_year_cls = (y_train_year > CONTACT_THRESHOLD).astype(np.float32)
        y_test_year_cls = (y_test_year > CONTACT_THRESHOLD).astype(np.float32)

        annual_indices = np.arange(len(y_train_year_cls))
        annual_train_indices, annual_val_indices = train_test_split(
            annual_indices,
            test_size=VAL_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=y_train_year_cls,
        )

        X_train_year = keep_last_n_pitches(X_train_year, MAX_PITCHES)
        X_test_year = keep_last_n_pitches(X_test_year, MAX_PITCHES)

        metrics_df, calibration_df, _ = train_evaluate_model(
            X_train=X_train_year,
            C_train=C_train_year,
            y_train_cls=y_train_year_cls,
            X_test=X_test_year,
            C_test=C_test_year,
            y_test_cls=y_test_year_cls,
            meta_test=meta_test_year,
            train_indices=annual_train_indices,
            val_indices=annual_val_indices,
            selected_params=selected_params,
            pitch_feature_columns=annual_pitch_feature_columns,
            c_feature_columns=annual_c_feature_columns,
            metadata={
                "model": model_name,
                "train_years": ",".join(map(str, train_years)),
                "test_year": int(test_year),
            },
            model_save_dir=EXPANDING_MODELS_DIR / model_name,
        )
        expanding_metric_tables.append(metrics_df)
        expanding_calibration_tables.append(calibration_df)

        del X_train_year, C_train_year, y_train_year
        del X_test_year, C_test_year, y_test_year
        gc.collect()

    # Reuse the Full-model evaluation as the final expanding-window row.
    full_as_expanding_metrics = full_metrics_df.copy()
    full_as_expanding_metrics["model"] = full_model_name
    full_as_expanding_calibration = full_calibration_df.copy()
    full_as_expanding_calibration["model"] = full_model_name
    expanding_metric_tables.append(full_as_expanding_metrics)
    expanding_calibration_tables.append(full_as_expanding_calibration)

    expanding_metrics_df = pd.concat(expanding_metric_tables, ignore_index=True)
    expanding_calibration_df = pd.concat(expanding_calibration_tables, ignore_index=True)
    expanding_metrics_df.to_csv(
        EXPANDING_WINDOW_DIR / "metrics_expanding_window.csv", index=False
    )
    expanding_calibration_df.to_csv(
        EXPANDING_WINDOW_DIR / "calibration_expanding_window.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 2025 subgroup analysis using the Full-model predictions.
    # Only two compact result files are written: metrics and calibration.
    # -------------------------------------------------------------------------
    subgroup_df = add_model_input_final_location(
        full_prediction_df,
        X_test_full,
        pitch_feature_columns,
    )
    subgroup_df = load_last_pitch_raw_metadata(
        prediction_df=subgroup_df,
        raw_data_root=RAW_PITCH_DATA_ROOT,
        test_year=2025,
    )

    subgroup_df["last_balls"] = pd.to_numeric(subgroup_df["last_balls"], errors="coerce")
    subgroup_df["count_group"] = subgroup_df["last_balls"].map({
        0.0: "0-2", 1.0: "1-2", 2.0: "2-2", 3.0: "3-2"
    })
    subgroup_df["strike_zone_group"] = subgroup_df.apply(
        classify_strike_zone_from_model_input, axis=1
    )
    subgroup_df["final_pitch_type_code"] = (
        subgroup_df["final_pitch_type_code"].astype("string").str.strip().str.upper()
    )
    subgroup_df["pitch_type_group"] = subgroup_df["final_pitch_type_code"].map(
        lambda code: f"{code}: {PITCH_TYPE_NAME_MAP.get(code, code)}"
        if pd.notna(code) else pd.NA
    )
    subgroup_df["batter_stand"] = (
        subgroup_df["batter_stand"].astype("string").str.strip().str.upper()
    )
    subgroup_df["batter_side_group"] = subgroup_df["batter_stand"].map(
        lambda hand: HAND_NAME_MAP.get(hand, f"Other ({hand})")
        if pd.notna(hand) else pd.NA
    )

    pitch_type_counts = subgroup_df["pitch_type_group"].dropna().value_counts()
    selected_pitch_types = pitch_type_counts[
        pitch_type_counts >= MIN_PITCH_TYPE_SAMPLES
    ].index.tolist()
    batter_side_order = [
        name for name in ["Left-handed batter", "Right-handed batter", "Switch hitter"]
        if name in set(subgroup_df["batter_side_group"].dropna().astype(str))
    ]

    subgroup_metric_tables = []
    subgroup_calibration_tables = []
    factor_definitions = [
        ("count", "count_group", ["0-2", "1-2", "2-2", "3-2"]),
        ("pitch_type", "pitch_type_group", selected_pitch_types),
        ("strike_zone", "strike_zone_group", ["Inside", "Outside"]),
        ("batter_side", "batter_side_group", batter_side_order),
    ]

    for factor_name, group_column, ordered_groups in factor_definitions:
        metrics_df, calibration_df = evaluate_subgroups(
            analysis_df=subgroup_df,
            group_column=group_column,
            ordered_groups=ordered_groups,
            factor_name=factor_name,
        )
        subgroup_metric_tables.append(metrics_df)
        subgroup_calibration_tables.append(calibration_df)

    # Pitcher K% tertiles.
    pitcher_stats_df = load_and_prepare_pitcher_stats(PITCHER_STATS_PATH)
    subgroup_df["pitcher"] = pd.to_numeric(
        subgroup_df["pitcher"], errors="coerce"
    ).astype("Int64")
    pitcher_analysis_df = subgroup_df.merge(
        pitcher_stats_df.rename(columns={
            "player_id": "pitcher",
            "Name": "pitcher_name_2025",
            "Team": "pitcher_team_2025",
        }),
        on="pitcher",
        how="left",
        validate="many_to_one",
    )
    matched_pitcher_df = pitcher_analysis_df[
        pitcher_analysis_df["pitcher_k_rate_2025"].notna()
    ].copy()
    assert not matched_pitcher_df.empty

    sample_coverage = len(matched_pitcher_df) / len(pitcher_analysis_df)
    unique_pitcher_coverage = (
        matched_pitcher_df["pitcher"].nunique()
        / max(pitcher_analysis_df["pitcher"].nunique(), 1)
    )
    assert sample_coverage >= EXPECTED_MIN_PITCHER_SAMPLE_COVERAGE

    group_definition_df = (
        pitcher_stats_df.groupby("pitcher_k_rate_group", observed=False)
        .agg(
            n_pitchers=("player_id", "size"),
            minimum_k_rate=("pitcher_k_rate_2025", "min"),
            median_k_rate=("pitcher_k_rate_2025", "median"),
            maximum_k_rate=("pitcher_k_rate_2025", "max"),
        )
        .reindex(PITCHER_K_GROUP_ORDER)
        .reset_index()
    )
    pitcher_group_metadata = {}
    for row in group_definition_df.itertuples(index=False):
        pitcher_group_metadata[str(row.pitcher_k_rate_group)] = {
            "n_pitchers_in_group": int(row.n_pitchers),
            "minimum_k_rate": float(row.minimum_k_rate),
            "median_k_rate": float(row.median_k_rate),
            "maximum_k_rate": float(row.maximum_k_rate),
            "pitcher_stats_sample_coverage": float(sample_coverage),
            "pitcher_stats_unique_pitcher_coverage": float(unique_pitcher_coverage),
        }

    pitcher_metrics_df, pitcher_calibration_df = evaluate_subgroups(
        analysis_df=matched_pitcher_df,
        group_column="pitcher_k_rate_group",
        ordered_groups=PITCHER_K_GROUP_ORDER,
        factor_name="pitcher_k_rate_tertile",
        extra_group_metadata=pitcher_group_metadata,
    )
    subgroup_metric_tables.append(pitcher_metrics_df)
    subgroup_calibration_tables.append(pitcher_calibration_df)

    subgroup_metrics_df = pd.concat(subgroup_metric_tables, ignore_index=True)
    subgroup_calibration_df = pd.concat(subgroup_calibration_tables, ignore_index=True)
    subgroup_metrics_df.insert(0, "test_year", 2025)
    subgroup_calibration_df.insert(0, "test_year", 2025)

    subgroup_metrics_df.to_csv(
        SUBGROUP_DIR / "metrics_subgroup_analysis.csv", index=False
    )
    subgroup_calibration_df.to_csv(
        SUBGROUP_DIR / "calibration_subgroup_analysis.csv", index=False
    )

    print("\nSaved outputs:")
    print(FULL_MODEL_DIR / "metrics_full_model.csv")
    print(FULL_MODEL_DIR / "calibration_full_model.csv")
    print(EXPANDING_WINDOW_DIR / "metrics_expanding_window.csv")
    print(EXPANDING_WINDOW_DIR / "calibration_expanding_window.csv")
    print(EXPANDING_MODELS_DIR)
    print(SUBGROUP_DIR / "metrics_subgroup_analysis.csv")
    print(SUBGROUP_DIR / "calibration_subgroup_analysis.csv")


if __name__ == "__main__":
    main()
