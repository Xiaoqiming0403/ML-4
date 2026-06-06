import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/wdbc.data")
OUTPUT_DIR = Path("outputs")
SEED = 42


FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def load_data():
    columns = ["id", "diagnosis"] + FEATURE_NAMES
    df = pd.read_csv(DATA_PATH, header=None, names=columns)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = (df["diagnosis"] == "M").astype(float).to_numpy()
    return X, y


def stratified_split(y, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return (
        rng.permutation(np.array(train_idx)),
        rng.permutation(np.array(val_idx)),
        rng.permutation(np.array(test_idx)),
    )


def prepare_data(small_train_size=None):
    X, y = load_data()
    train_idx, val_idx, test_idx = stratified_split(y)

    if small_train_size is not None:
        rng = np.random.default_rng(SEED)
        selected = []
        for label in np.unique(y[train_idx]):
            label_idx = train_idx[y[train_idx] == label]
            k = max(1, int(small_train_size * len(label_idx) / len(train_idx)))
            selected.extend(rng.permutation(label_idx)[:k])
        train_idx = rng.permutation(np.array(selected))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.column_stack([X_train, np.ones(len(X_train))])
    X_val = np.column_stack([X_val, np.ones(len(X_val))])
    X_test = np.column_stack([X_test, np.ones(len(X_test))])
    return X_train, y_train, X_val, y_val, X_test, y_test


def loss_and_grad_z(name, z, y):
    p = sigmoid(z)
    y_pm = 2 * y - 1
    margin = y_pm * z
    eps = 1e-15

    if name == "logistic_bce":
        loss = -(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps)))
        grad_z = p - y
    elif name == "linear_probability":
        loss = y * (1 - p) + (1 - y) * p
        grad_z = (1 - 2 * y) * p * (1 - p)
    elif name == "squared_probability":
        loss = 0.5 * (p - y) ** 2
        grad_z = (p - y) * p * (1 - p)
    elif name == "quartic_probability":
        loss = 0.25 * (p - y) ** 4
        grad_z = (p - y) ** 3 * p * (1 - p)
    elif name == "exponential_margin":
        safe_margin = np.clip(margin, -20, 20)
        loss = np.exp(-safe_margin)
        grad_z = -y_pm * np.exp(-safe_margin)
    elif name == "squared_hinge_margin":
        h = np.maximum(0, 1 - margin)
        loss = h ** 2
        grad_z = -2 * y_pm * h
    elif name == "cubic_hinge_margin":
        h = np.maximum(0, 1 - margin)
        loss = h ** 3
        grad_z = -3 * y_pm * h ** 2
    elif name == "log_cosh_margin":
        r = np.clip(1 - margin, -20, 20)
        loss = np.log(np.cosh(r))
        grad_z = -y_pm * np.tanh(r)
    else:
        raise ValueError(f"Unknown loss: {name}")

    return loss, grad_z


LOSS_CONFIGS = [
    {"name": "logistic_bce", "lr": 0.08, "epochs": 500, "l2": 0.01},
    {"name": "linear_probability", "lr": 0.2, "epochs": 500, "l2": 0.01},
    {"name": "squared_probability", "lr": 0.2, "epochs": 500, "l2": 0.01},
    {"name": "quartic_probability", "lr": 0.5, "epochs": 500, "l2": 0.01},
    {"name": "exponential_margin", "lr": 0.002, "epochs": 500, "l2": 0.01},
    {"name": "squared_hinge_margin", "lr": 0.01, "epochs": 500, "l2": 0.01},
    {"name": "cubic_hinge_margin", "lr": 0.002, "epochs": 500, "l2": 0.01},
    {"name": "log_cosh_margin", "lr": 0.04, "epochs": 500, "l2": 0.01},
]


DISPLAY_NAMES = {
    "logistic_bce": "Logistic BCE",
    "linear_probability": "Linear probability loss",
    "squared_probability": "Squared probability loss",
    "quartic_probability": "Quartic probability loss",
    "exponential_margin": "Exponential margin loss",
    "squared_hinge_margin": "Squared hinge margin loss",
    "cubic_hinge_margin": "Cubic hinge margin loss",
    "log_cosh_margin": "Log-cosh margin loss",
}


def objective(X, y, w, loss_name, l2=0.0):
    losses, _ = loss_and_grad_z(loss_name, X @ w, y)
    return float(np.mean(losses) + l2 * np.sum(w[:-1] ** 2) / (2 * len(y)))


def train(X_train, y_train, X_val, y_val, config):
    rng = np.random.default_rng(SEED)
    n, d = X_train.shape
    w = np.zeros(d)
    losses = []
    val_losses = []
    batch_size = 32
    start = time.perf_counter()

    for _ in range(config["epochs"]):
        idx = rng.permutation(n)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        for i in range(0, n, batch_size):
            Xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            _, grad_z = loss_and_grad_z(config["name"], Xb @ w, yb)
            grad_w = Xb.T @ grad_z / len(yb)
            grad_w[:-1] += config["l2"] * w[:-1] / len(yb)
            grad_w = np.clip(grad_w, -50, 50)
            w -= config["lr"] * grad_w

        losses.append(objective(X_train, y_train, w, config["name"], config["l2"]))
        val_losses.append(objective(X_val, y_val, w, config["name"], config["l2"]))

    seconds = time.perf_counter() - start
    return w, losses, val_losses, seconds


def roc_auc(y_true, y_score):
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1
    for threshold in np.linspace(0.05, 0.95, 91):
        f1 = metrics(y_true, y_prob, threshold)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(threshold), f1
    return best_t


def run_one_setting(setting_name, small_train_size=None):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(small_train_size=small_train_size)
    rows = []
    loss_curves = {}
    for config in LOSS_CONFIGS:
        w, train_losses, val_losses, seconds = train(X_train, y_train, X_val, y_val, config)
        val_prob = sigmoid(X_val @ w)
        test_prob = sigmoid(X_test @ w)
        threshold = best_threshold(y_val, val_prob)
        row = {
            "setting": setting_name,
            "loss": config["name"],
            "display_name": DISPLAY_NAMES[config["name"]],
            "lr": config["lr"],
            "epochs": config["epochs"],
            "l2": config["l2"],
            "train_size": len(y_train),
            "time_seconds": seconds,
            "final_train_objective": train_losses[-1],
            "final_val_objective": val_losses[-1],
            "threshold": threshold,
        }
        test_metrics = metrics(y_test, test_prob, threshold)
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        rows.append(row)
        loss_curves[f"{setting_name}: {DISPLAY_NAMES[config['name']]}"] = train_losses
    return rows, loss_curves


def save_loss_svg(loss_dict, path):
    width, height = 1000, 620
    margin = 70
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#475569"]
    all_values = [v for losses in loss_dict.values() for v in losses if math.isfinite(v)]
    min_v, max_v = min(all_values), max(all_values)
    span = max(max_v - min_v, 1e-12)

    def xy(i, value, n):
        x = margin + i * (width - 2 * margin) / max(n - 1, 1)
        y = height - margin - (value - min_v) * (height - 2 * margin) / span
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="34" text-anchor="middle" font-size="22" font-family="Arial">Training Objective Curves</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111"/>',
    ]
    for idx, (name, losses) in enumerate(loss_dict.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x:.2f},{y:.2f}" for i, value in enumerate(losses) for x, y in [xy(i, value, len(losses))])
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{points}"/>')
        parts.append(f'<text x="{width - margin + 10}" y="{margin + 18 * idx}" font-size="11" font-family="Arial" fill="{color}">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_rows, full_curves = run_one_setting("full_train", small_train_size=None)
    small_rows, small_curves = run_one_setting("small_train_80", small_train_size=80)
    rows = full_rows + small_rows

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "loss_function_comparison.csv", index=False)
    save_loss_svg(full_curves, OUTPUT_DIR / "full_train_loss_curves.svg")
    save_loss_svg(small_curves, OUTPUT_DIR / "small_train_loss_curves.svg")

    columns = [
        "setting", "display_name", "train_size", "threshold", "test_accuracy",
        "test_precision", "test_recall", "test_f1", "test_auc",
        "final_train_objective", "time_seconds",
    ]
    summary = [
        "# Breast Cancer Custom Loss Function Experiments",
        "",
        "## Results",
        "",
        markdown_table(rows, columns),
    ]
    (OUTPUT_DIR / "loss_results_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    run()
