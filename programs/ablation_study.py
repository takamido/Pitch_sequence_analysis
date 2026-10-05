import gc
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

DATA_ROOT = Path("./dataset/last")

OUTPUT_ROOT = Path("./Ablation_study")
ABLATION_OUTPUT_DIR = OUTPUT_ROOT / "ablation"
BASELINE_OUTPUT_DIR = OUTPUT_ROOT / "baseline"

ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Full-model architecture reported in Section 2.3.
SELECTED_PARAMS = {
    "d_model": 256,
    "num_heads": 4,
    "ff_dim": 512,
    "num_blocks": 2,
    "dense_units": 64,
}

WINDOW_CONFIGS = [
    {"train_tag": "2018_19", "train_years": [2018, 2019], "test_year": 2022},
    {"train_tag": "2018_22", "train_years": [2018, 2019, 2022], "test_year": 2023},
    {"train_tag": "2018_23", "train_years": [2018, 2019, 2022, 2023], "test_year": 2024},
    {"train_tag": "2018_24", "train_years": [2018, 2019, 2022, 2023, 2024], "test_year": 2025},
]

SEQUENCE_VARIANT_NAMES = [
    *[f"last_{n_pitches}" for n_pitches in range(1, MAX_PITCHES + 1)],
    *[f"last_{n_from_last}_only" for n_from_last in range(2, MAX_PITCHES + 1)],
    "without_final_pitch",
    "shuffled_valid_pitches",
    "shuffled_older_keep_final",
    "final_plus_random_older",
]
VARIANT_ORDER = [*SEQUENCE_VARIANT_NAMES, "context_only"]
REFERENCE_VARIANT = "last_6"

METRICS_FOR_CI = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "brier_score",
    "ece",
]


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
    return (mean.astype(np.float32), std.astype(np.float32))


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
    return (mean.astype(np.float32), std.astype(np.float32))


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


def shuffle_valid_pitches_keep_padding(X, seed):
    """Shuffle all valid pitches while keeping padding positions unchanged."""
    rng = np.random.default_rng(seed)
    transformed = X.copy()
    for sample_idx, sequence in enumerate(X):
        positions = np.flatnonzero(valid_pitch_mask(sequence))
        permutation = rng.permutation(len(positions))
        transformed[sample_idx, positions] = sequence[positions][permutation]
    return transformed


def shuffle_older_pitches_keep_final(X, seed):
    """Shuffle only pitches before the final pitch; keep the final pitch fixed."""
    rng = np.random.default_rng(seed)
    transformed = X.copy()
    for sample_idx, sequence in enumerate(X):
        positions = np.flatnonzero(valid_pitch_mask(sequence))
        if len(positions) <= 2:
            continue
        older_positions = positions[:-1]
        permutation = rng.permutation(len(older_positions))
        transformed[sample_idx, older_positions] = sequence[older_positions][permutation]
    return transformed


def keep_nth_from_last_pitch_only(X, n_from_last):
    """Keep only the n-th pitch from the end at its original sequence position.

    n_from_last=1 is the final pitch, 2 is the second-to-last pitch, etc.
    If a plate appearance does not contain that many valid pitches, its pitch
    sequence is left as all padding so the plate-appearance set stays fixed.
    """
    n_from_last = int(n_from_last)
    if n_from_last < 1:
        raise ValueError('n_from_last must be >= 1')
    transformed = np.zeros_like(X)
    for sample_idx, sequence in enumerate(X):
        positions = np.flatnonzero(valid_pitch_mask(sequence))
        if len(positions) < n_from_last:
            continue
        selected_position = positions[-n_from_last]
        transformed[sample_idx, selected_position] = sequence[selected_position]
    return transformed


def final_plus_random_older_pitch(X, seed):
    rng = np.random.default_rng(seed)
    transformed = np.zeros_like(X)
    for sample_idx, sequence in enumerate(X):
        valid = sequence[valid_pitch_mask(sequence)]
        final_pitch = valid[-1]
        older_candidates = valid[:-2]
        chosen_idx = rng.integers(0, len(older_candidates))
        transformed[sample_idx, -2] = older_candidates[chosen_idx]
        transformed[sample_idx, -1] = final_pitch
    return transformed


def remove_final_pitch_keep_older(X):
    transformed = X.copy()
    for sample_idx, sequence in enumerate(X):
        positions = np.flatnonzero(valid_pitch_mask(sequence))
        transformed[sample_idx, positions[-1]] = 0
    return transformed


def build_sequence_variant(X, variant_name, seed):
    if variant_name == 'without_final_pitch':
        return remove_final_pitch_keep_older(X)
    only_match = re.fullmatch('last_([2-6])_only', variant_name)
    if only_match:
        return keep_nth_from_last_pitch_only(X, n_from_last=int(only_match.group(1)))
    cumulative_match = re.fullmatch('last_([1-6])', variant_name)
    if cumulative_match:
        return keep_last_n_pitches(X, int(cumulative_match.group(1)))
    if variant_name == 'shuffled_valid_pitches':
        return shuffle_valid_pitches_keep_padding(X, seed)
    if variant_name == 'shuffled_older_keep_final':
        return shuffle_older_pitches_keep_final(X, seed)
    if variant_name == 'final_plus_random_older':
        return final_plus_random_older_pitch(X, seed)
    raise ValueError(f'Unknown sequence variant: {variant_name}')


@tf.keras.utils.register_keras_serializable(package='PitchSequence')
class PaddingMaskLayer(layers.Layer):

    def call(self, inputs):
        is_padding = tf.reduce_all(tf.equal(inputs, 0.0), axis=-1)
        return tf.logical_not(is_padding)


@tf.keras.utils.register_keras_serializable(package='PitchSequence')
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
        config.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return config


@tf.keras.utils.register_keras_serializable(package='PitchSequence')
class TransformerEncoderBlock(layers.Layer):

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization()
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dropout(dropout_rate), layers.Dense(d_model)])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x, valid_mask = inputs
        attn_mask = tf.cast(valid_mask[:, tf.newaxis, :], tf.bool)
        attn_output = self.attn(x, x, attention_mask=attn_mask, training=training)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'dropout_rate': self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package='PitchSequence')
class MaskedMeanPooling(layers.Layer):

    def call(self, inputs):
        x, valid_mask = inputs
        mask = tf.cast(valid_mask, tf.float32)[:, :, tf.newaxis]
        sum_x = tf.reduce_sum(x * mask, axis=1)
        count = tf.reduce_sum(mask, axis=1)
        return sum_x / tf.maximum(count, 1.0)


def compile_binary_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model


def build_transformer_classifier_with_context(pitch_input_shape, context_input_shape, d_model=128, num_heads=4, ff_dim=256, num_blocks=2, context_dense_units=32, dense_units=64, dropout_rate=0.0, learning_rate=0.001):
    pitch_inputs = layers.Input(shape=pitch_input_shape, name='pitch_sequence')
    valid_mask = PaddingMaskLayer(name='padding_mask')(pitch_inputs)
    x = layers.Dense(d_model, name='pitch_projection')(pitch_inputs)
    x = PositionalEmbedding(seq_len=pitch_input_shape[0], d_model=d_model, name='positional_embedding')(x)
    for block_idx in range(num_blocks):
        x = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, name=f'transformer_encoder_block_{block_idx + 1}')([x, valid_mask])
    pitch_repr = MaskedMeanPooling(name='masked_mean_pooling')([x, valid_mask])
    pitch_repr = layers.Dense(dense_units, activation='relu', name='pitch_repr_dense')(pitch_repr)
    pitch_repr = layers.Dropout(dropout_rate, name='pitch_repr_dropout')(pitch_repr)
    context_inputs = layers.Input(shape=context_input_shape, name='context_features')
    context_repr = layers.Dense(context_dense_units, activation='relu', name='context_repr_dense')(context_inputs)
    context_repr = layers.Dropout(dropout_rate, name='context_repr_dropout')(context_repr)
    merged = layers.Concatenate(name='pitch_context_concat')([pitch_repr, context_repr])
    z = layers.Dense(dense_units, activation='relu', name='merged_dense')(merged)
    z = layers.Dropout(dropout_rate, name='merged_dropout')(z)
    outputs = layers.Dense(1, activation='sigmoid', name='in_play_probability')(z)
    model = models.Model(inputs=[pitch_inputs, context_inputs], outputs=outputs, name='transformer_binary_classifier_with_context_C')
    return compile_binary_model(model, learning_rate)


def build_context_only_classifier(context_input_shape, context_dense_units=32, dense_units=64, dropout_rate=0.0, learning_rate=0.001):
    context_inputs = layers.Input(shape=context_input_shape, name='context_features')
    x = layers.Dense(context_dense_units, activation='relu', name='context_repr_dense')(context_inputs)
    x = layers.Dropout(dropout_rate, name='context_repr_dropout')(x)
    x = layers.Dense(dense_units, activation='relu', name='context_dense')(x)
    x = layers.Dropout(dropout_rate, name='context_dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='in_play_probability')(x)
    model = models.Model(inputs=context_inputs, outputs=outputs, name='context_only_binary_classifier')
    return compile_binary_model(model, learning_rate)


def make_model_inputs(uses_sequence, X_scaled, C_scaled):
    if uses_sequence:
        return {'pitch_sequence': X_scaled, 'context_features': C_scaled}
    return {'context_features': C_scaled}


def probability_to_logit(y_prob, eps=1e-06):
    """Convert sigmoid probabilities to finite logits for Platt scaling."""
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return np.log(y_prob / (1.0 - y_prob))


def fit_platt_scaler(y_val_true, y_val_prob_raw):
    """Fit Platt scaling on validation labels and raw model probabilities only."""
    y_val_true = np.asarray(y_val_true).astype(int).ravel()
    val_logits = probability_to_logit(y_val_prob_raw).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
    calibrator.fit(val_logits, y_val_true)
    return calibrator


def apply_platt_scaler(calibrator, y_prob_raw):
    logits = probability_to_logit(y_prob_raw).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1].astype(np.float64)


set_global_seed(RANDOM_SEED)


def generate_unique_bootstrap_seeds(n_bootstrap, base_seed):
    child_sequences = np.random.SeedSequence(int(base_seed)).spawn(int(n_bootstrap))
    return np.asarray(
        [
            int(child_sequence.generate_state(1, dtype=np.uint32)[0])
            for child_sequence in child_sequences
        ],
        dtype=np.uint64,
    )


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
            mean_predicted_probability = float(y_prob[mask].mean())
            observed_positive_rate = float(y_true[mask].mean())
            absolute_calibration_gap = abs(
                observed_positive_rate - mean_predicted_probability
            )
            sample_fraction = count / n_samples
            ece += sample_fraction * absolute_calibration_gap
        else:
            mean_predicted_probability = np.nan
            observed_positive_rate = np.nan
            absolute_calibration_gap = np.nan
            sample_fraction = 0.0

        rows.append(
            {
                "bin_index": bin_idx + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "sample_fraction": float(sample_fraction),
                "mean_predicted_probability": mean_predicted_probability,
                "observed_positive_rate": observed_positive_rate,
                "absolute_calibration_gap": absolute_calibration_gap,
            }
        )

    return brier_score, float(ece), pd.DataFrame(rows)


def calculate_point_metrics(y_true, y_prob, threshold=CLASSIFICATION_THRESHOLD):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc = (
        float(roc_auc_score(y_true, y_prob))
        if np.unique(y_true).size >= 2
        else np.nan
    )
    brier_score, ece, _ = calculate_calibration_statistics(
        y_true, y_prob, n_bins=CALIBRATION_N_BINS
    )

    return {
        "n_samples": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "negative_n": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(y_prob.mean()),
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


def evaluate_with_bootstrap(
    y_true,
    y_prob,
    n_bootstrap=BOOTSTRAP_N,
    base_seed=RANDOM_SEED,
    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    point_metrics = calculate_point_metrics(y_true, y_prob)
    _, _, point_calibration = calculate_calibration_statistics(
        y_true, y_prob, n_bins=CALIBRATION_N_BINS
    )

    bootstrap_seeds = generate_unique_bootstrap_seeds(n_bootstrap, base_seed)
    iteration_rows = []
    calibration_rows = []
    sample_size = len(y_true)

    for bootstrap_id, bootstrap_seed in enumerate(bootstrap_seeds, start=1):
        rng = np.random.default_rng(int(bootstrap_seed))
        sampled_indices = rng.integers(
            low=0,
            high=sample_size,
            size=sample_size,
            endpoint=False,
        )

        sampled_y_true = y_true[sampled_indices]
        sampled_y_prob = y_prob[sampled_indices]

        replicate_metrics = calculate_point_metrics(
            sampled_y_true, sampled_y_prob
        )
        iteration_rows.append(
            {
                "bootstrap_id": int(bootstrap_id),
                "bootstrap_seed": int(bootstrap_seed),
                **replicate_metrics,
            }
        )

        _, _, replicate_calibration = calculate_calibration_statistics(
            sampled_y_true,
            sampled_y_prob,
            n_bins=CALIBRATION_N_BINS,
        )
        replicate_calibration["bootstrap_id"] = int(bootstrap_id)
        replicate_calibration["bootstrap_seed"] = int(bootstrap_seed)
        calibration_rows.append(replicate_calibration)

    iteration_df = pd.DataFrame(iteration_rows)
    calibration_iteration_df = pd.concat(calibration_rows, ignore_index=True)

    alpha = (1.0 - float(confidence_level)) / 2.0

    metrics_summary = dict(point_metrics)
    metrics_summary["bootstrap_n"] = int(n_bootstrap)
    metrics_summary["confidence_level"] = float(confidence_level)

    for metric_name in METRICS_FOR_CI:
        values = pd.to_numeric(
            iteration_df[metric_name], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        metrics_summary[f"{metric_name}_ci_lower"] = (
            float(np.quantile(values, alpha)) if len(values) else np.nan
        )
        metrics_summary[f"{metric_name}_ci_upper"] = (
            float(np.quantile(values, 1.0 - alpha)) if len(values) else np.nan
        )

    calibration_summary_rows = []
    for point_row in point_calibration.itertuples(index=False):
        group_df = calibration_iteration_df[
            calibration_iteration_df["bin_index"] == point_row.bin_index
        ]

        summary_row = {
            "bin_index": int(point_row.bin_index),
            "bin_lower": float(point_row.bin_lower),
            "bin_upper": float(point_row.bin_upper),
            "count": int(point_row.count),
            "sample_fraction": float(point_row.sample_fraction),
            "mean_predicted_probability": point_row.mean_predicted_probability,
            "observed_positive_rate": point_row.observed_positive_rate,
            "absolute_calibration_gap": point_row.absolute_calibration_gap,
            "confidence_level": float(confidence_level),
            "bootstrap_n": int(n_bootstrap),
        }

        for column in [
            "count",
            "sample_fraction",
            "mean_predicted_probability",
            "observed_positive_rate",
            "absolute_calibration_gap",
        ]:
            values = pd.to_numeric(
                group_df[column], errors="coerce"
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            summary_row[f"{column}_ci_lower"] = (
                float(np.quantile(values, alpha)) if len(values) else np.nan
            )
            summary_row[f"{column}_ci_upper"] = (
                float(np.quantile(values, 1.0 - alpha))
                if len(values)
                else np.nan
            )

        calibration_summary_rows.append(summary_row)

    return (
        metrics_summary,
        pd.DataFrame(calibration_summary_rows),
        iteration_df,
    )


def train_transformer_variant(
    variant_name,
    uses_sequence,
    X_variant_train,
    X_variant_test,
    C_train,
    C_test,
    y_train_cls,
    y_test_cls,
    train_indices,
    val_indices,
):
    C_tr = C_train[train_indices]
    C_val = C_train[val_indices]
    y_tr = y_train_cls[train_indices]
    y_val = y_train_cls[val_indices]

    c_mean, c_std = fit_tabular_standardizer(C_tr)
    C_tr_scaled = transform_tabular_standardizer(C_tr, c_mean, c_std)
    C_val_scaled = transform_tabular_standardizer(C_val, c_mean, c_std)
    C_test_scaled = transform_tabular_standardizer(C_test, c_mean, c_std)

    if uses_sequence:
        X_tr = X_variant_train[train_indices]
        X_val = X_variant_train[val_indices]
        x_mean, x_std = fit_sequence_standardizer(X_tr)
        X_tr_scaled = transform_sequence_standardizer(X_tr, x_mean, x_std)
        X_val_scaled = transform_sequence_standardizer(X_val, x_mean, x_std)
        X_test_scaled = transform_sequence_standardizer(
            X_variant_test, x_mean, x_std
        )
    else:
        X_tr_scaled = X_val_scaled = X_test_scaled = None

    tf.keras.backend.clear_session()
    set_global_seed(RANDOM_SEED)

    if uses_sequence:
        model = build_transformer_classifier_with_context(
            pitch_input_shape=X_tr_scaled.shape[1:],
            context_input_shape=C_tr_scaled.shape[1:],
            d_model=SELECTED_PARAMS["d_model"],
            num_heads=SELECTED_PARAMS["num_heads"],
            ff_dim=SELECTED_PARAMS["ff_dim"],
            num_blocks=SELECTED_PARAMS["num_blocks"],
            context_dense_units=CONTEXT_DENSE_UNITS,
            dense_units=SELECTED_PARAMS["dense_units"],
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
        )
    else:
        model = build_context_only_classifier(
            context_input_shape=C_tr_scaled.shape[1:],
            context_dense_units=CONTEXT_DENSE_UNITS,
            dense_units=SELECTED_PARAMS["dense_units"],
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
        )

    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        make_model_inputs(uses_sequence, X_tr_scaled, C_tr_scaled),
        y_tr,
        validation_data=(
            make_model_inputs(uses_sequence, X_val_scaled, C_val_scaled),
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_cb],
        verbose=1,
    )

    y_val_prob_raw = model.predict(
        make_model_inputs(uses_sequence, X_val_scaled, C_val_scaled),
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()

    platt_scaler = fit_platt_scaler(y_val, y_val_prob_raw)

    y_test_prob_raw = model.predict(
        make_model_inputs(uses_sequence, X_test_scaled, C_test_scaled),
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()
    y_test_prob = apply_platt_scaler(platt_scaler, y_test_prob_raw)

    metrics_summary, calibration_summary, bootstrap_iterations = (
        evaluate_with_bootstrap(
            y_true=y_test_cls,
            y_prob=y_test_prob,
            n_bootstrap=BOOTSTRAP_N,
            base_seed=RANDOM_SEED,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        )
    )

    metrics_summary.update(
        {
            "variant": variant_name,
            "probability_version": "platt",
            "uses_sequence": bool(uses_sequence),
            "trained_epochs": int(len(history.history["loss"])),
            **SELECTED_PARAMS,
        }
    )

    calibration_summary.insert(0, "variant", variant_name)
    calibration_summary.insert(1, "probability_version", "platt")

    del model, platt_scaler
    tf.keras.backend.clear_session()

    return metrics_summary, calibration_summary, bootstrap_iterations


def evaluate_baseline(y_true, y_prob, baseline_name):
    metrics_summary, calibration_summary, bootstrap_iterations = (
        evaluate_with_bootstrap(
            y_true=y_true,
            y_prob=y_prob,
            n_bootstrap=BOOTSTRAP_N,
            base_seed=RANDOM_SEED,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        )
    )

    metrics_summary.update(
        {
            "variant": baseline_name,
            "probability_version": "raw",
        }
    )
    calibration_summary.insert(0, "variant", baseline_name)
    calibration_summary.insert(1, "probability_version", "raw")

    return metrics_summary, calibration_summary, bootstrap_iterations


def add_paired_differences_vs_reference(
    metrics_df,
    bootstrap_iterations_by_variant,
    reference_variant=REFERENCE_VARIANT,
):
    metrics_df = metrics_df.copy()
    reference_iterations = bootstrap_iterations_by_variant[reference_variant]

    alpha = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0

    for metric_name in METRICS_FOR_CI:
        metrics_df[f"{metric_name}_difference_vs_{reference_variant}"] = np.nan
        metrics_df[f"{metric_name}_difference_ci_lower_vs_{reference_variant}"] = np.nan
        metrics_df[f"{metric_name}_difference_ci_upper_vs_{reference_variant}"] = np.nan

    reference_point_row = metrics_df[
        metrics_df["variant"] == reference_variant
    ].iloc[0]

    for row_index, row in metrics_df.iterrows():
        variant_name = row["variant"]

        if variant_name == reference_variant:
            for metric_name in METRICS_FOR_CI:
                metrics_df.loc[
                    row_index,
                    f"{metric_name}_difference_vs_{reference_variant}",
                ] = 0.0
                metrics_df.loc[
                    row_index,
                    f"{metric_name}_difference_ci_lower_vs_{reference_variant}",
                ] = 0.0
                metrics_df.loc[
                    row_index,
                    f"{metric_name}_difference_ci_upper_vs_{reference_variant}",
                ] = 0.0
            continue

        candidate_iterations = bootstrap_iterations_by_variant[variant_name]
        paired = candidate_iterations.merge(
            reference_iterations,
            on=["bootstrap_id", "bootstrap_seed"],
            how="inner",
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        )

        for metric_name in METRICS_FOR_CI:
            point_difference = (
                float(row[metric_name]) - float(reference_point_row[metric_name])
            )
            differences = (
                paired[f"{metric_name}_candidate"]
                - paired[f"{metric_name}_reference"]
            ).to_numpy(dtype=float)
            differences = differences[np.isfinite(differences)]

            metrics_df.loc[
                row_index,
                f"{metric_name}_difference_vs_{reference_variant}",
            ] = point_difference
            metrics_df.loc[
                row_index,
                f"{metric_name}_difference_ci_lower_vs_{reference_variant}",
            ] = (
                float(np.quantile(differences, alpha))
                if len(differences)
                else np.nan
            )
            metrics_df.loc[
                row_index,
                f"{metric_name}_difference_ci_upper_vs_{reference_variant}",
            ] = (
                float(np.quantile(differences, 1.0 - alpha))
                if len(differences)
                else np.nan
            )

    return metrics_df


def run_one_test_year(config):
    train_tag = config["train_tag"]
    train_years = config["train_years"]
    test_year = int(config["test_year"])

    print("\n" + "#" * 100)
    print(f"Test year: {test_year}")
    print(f"Training years: {train_years}")

    dataset_path = DATA_ROOT / f"dataset_last_{train_tag}.npz"

    with np.load(dataset_path, allow_pickle=True) as data:
        X_train = data["X_train"].astype(np.float32)
        C_train = data["C_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        C_test = data["C_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)

    assert X_train.shape[1] == X_test.shape[1] == MAX_PITCHES

    y_train_cls = (y_train > CONTACT_THRESHOLD).astype(np.float32)
    y_test_cls = (y_test > CONTACT_THRESHOLD).astype(np.float32)

    all_train_indices = np.arange(len(y_train_cls))
    train_indices, val_indices = train_test_split(
        all_train_indices,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_train_cls,
    )

    common_metadata = {
        "train_years": ",".join(map(str, train_years)),
        "test_year": test_year,
        "n_train_total": int(len(y_train_cls)),
        "n_train_fit": int(len(train_indices)),
        "n_validation": int(len(val_indices)),
        "n_test": int(len(y_test_cls)),
    }

    # -------------------------
    # Ablation models
    # -------------------------
    ablation_metric_rows = []
    ablation_calibration_tables = []
    bootstrap_iterations_by_variant = {}

    for variant_name in VARIANT_ORDER:
        print(f"\nAblation variant: {variant_name}")

        uses_sequence = variant_name != "context_only"
        if uses_sequence:
            X_variant_train = build_sequence_variant(
                X_train, variant_name, seed=RANDOM_SEED
            )
            X_variant_test = build_sequence_variant(
                X_test, variant_name, seed=RANDOM_SEED + 1
            )
        else:
            X_variant_train = None
            X_variant_test = None

        metric_row, calibration_df, bootstrap_iterations = (
            train_transformer_variant(
                variant_name=variant_name,
                uses_sequence=uses_sequence,
                X_variant_train=X_variant_train,
                X_variant_test=X_variant_test,
                C_train=C_train,
                C_test=C_test,
                y_train_cls=y_train_cls,
                y_test_cls=y_test_cls,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

        metric_row.update(common_metadata)
        calibration_df["train_years"] = common_metadata["train_years"]
        calibration_df["test_year"] = test_year

        ablation_metric_rows.append(metric_row)
        ablation_calibration_tables.append(calibration_df)
        bootstrap_iterations_by_variant[variant_name] = bootstrap_iterations

        if uses_sequence:
            del X_variant_train, X_variant_test
        gc.collect()

    metrics_ablation_df = pd.DataFrame(ablation_metric_rows)
    metrics_ablation_df["variant"] = pd.Categorical(
        metrics_ablation_df["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )
    metrics_ablation_df = metrics_ablation_df.sort_values(
        "variant"
    ).reset_index(drop=True)
    metrics_ablation_df["variant"] = metrics_ablation_df["variant"].astype(str)

    metrics_ablation_df = add_paired_differences_vs_reference(
        metrics_df=metrics_ablation_df,
        bootstrap_iterations_by_variant=bootstrap_iterations_by_variant,
        reference_variant=REFERENCE_VARIANT,
    )

    calibration_ablation_df = pd.concat(
        ablation_calibration_tables, ignore_index=True
    )

    metrics_ablation_df.to_csv(
        ABLATION_OUTPUT_DIR / f"metrics_ablation_{test_year}test.csv",
        index=False,
    )
    calibration_ablation_df.to_csv(
        ABLATION_OUTPUT_DIR / f"calibration_ablation_{test_year}test.csv",
        index=False,
    )

    # -------------------------
    # Baseline models
    # -------------------------
    baseline_metric_rows = []
    baseline_calibration_tables = []

    constant_probability = float(y_train_cls[train_indices].mean())
    constant_test_prob = np.full(
        len(y_test_cls),
        constant_probability,
        dtype=np.float64,
    )
    constant_metrics, constant_calibration, _ = evaluate_baseline(
        y_true=y_test_cls,
        y_prob=constant_test_prob,
        baseline_name="constant_baseline",
    )
    constant_metrics["training_positive_rate"] = constant_probability
    constant_metrics.update(common_metadata)
    constant_calibration["train_years"] = common_metadata["train_years"]
    constant_calibration["test_year"] = test_year

    baseline_metric_rows.append(constant_metrics)
    baseline_calibration_tables.append(constant_calibration)

    X_lr_train = keep_last_n_pitches(X_train, MAX_PITCHES)
    X_lr_test = keep_last_n_pitches(X_test, MAX_PITCHES)
    X_lr_tr = X_lr_train[train_indices]
    C_lr_tr = C_train[train_indices]
    y_lr_tr = y_train_cls[train_indices]

    x_lr_mean, x_lr_std = fit_sequence_standardizer(X_lr_tr)
    X_lr_tr_scaled = transform_sequence_standardizer(
        X_lr_tr, x_lr_mean, x_lr_std
    )
    X_lr_test_scaled = transform_sequence_standardizer(
        X_lr_test, x_lr_mean, x_lr_std
    )

    c_lr_mean, c_lr_std = fit_tabular_standardizer(C_lr_tr)
    C_lr_tr_scaled = transform_tabular_standardizer(
        C_lr_tr, c_lr_mean, c_lr_std
    )
    C_lr_test_scaled = transform_tabular_standardizer(
        C_test, c_lr_mean, c_lr_std
    )

    lr_tr_inputs = np.concatenate(
        [
            X_lr_tr_scaled.reshape(len(X_lr_tr_scaled), -1),
            C_lr_tr_scaled,
        ],
        axis=1,
    )
    lr_test_inputs = np.concatenate(
        [
            X_lr_test_scaled.reshape(len(X_lr_test_scaled), -1),
            C_lr_test_scaled,
        ],
        axis=1,
    )

    logistic_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    logistic_model.fit(lr_tr_inputs, y_lr_tr.astype(int))
    logistic_test_prob = logistic_model.predict_proba(lr_test_inputs)[:, 1]

    logistic_metrics, logistic_calibration, _ = evaluate_baseline(
        y_true=y_test_cls,
        y_prob=logistic_test_prob,
        baseline_name="logistic_last_6_context",
    )
    logistic_metrics.update(common_metadata)
    logistic_calibration["train_years"] = common_metadata["train_years"]
    logistic_calibration["test_year"] = test_year

    baseline_metric_rows.append(logistic_metrics)
    baseline_calibration_tables.append(logistic_calibration)

    metrics_baseline_df = pd.DataFrame(baseline_metric_rows)
    calibration_baseline_df = pd.concat(
        baseline_calibration_tables, ignore_index=True
    )

    metrics_baseline_df.to_csv(
        BASELINE_OUTPUT_DIR / f"metrics_baseline_{test_year}test.csv",
        index=False,
    )
    calibration_baseline_df.to_csv(
        BASELINE_OUTPUT_DIR / f"calibration_baseline_{test_year}test.csv",
        index=False,
    )

    del (
        X_train,
        C_train,
        y_train,
        X_test,
        C_test,
        y_test,
        y_train_cls,
        y_test_cls,
        X_lr_train,
        X_lr_test,
        X_lr_tr,
        X_lr_tr_scaled,
        X_lr_test_scaled,
        C_lr_tr,
        C_lr_tr_scaled,
        C_lr_test_scaled,
        lr_tr_inputs,
        lr_test_inputs,
        logistic_model,
    )
    gc.collect()


for config in WINDOW_CONFIGS:
    run_one_test_year(config)

print("\n===== Ablation study complete =====")
print("Output root:", OUTPUT_ROOT.resolve())
print("Ablation files:", ABLATION_OUTPUT_DIR.resolve())
print("Baseline files:", BASELINE_OUTPUT_DIR.resolve())
