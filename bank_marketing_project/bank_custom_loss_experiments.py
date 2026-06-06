import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/bank_marketing/bank-additional/bank-additional-full.csv")
OUTPUT_DIR = Path("outputs/custom_losses")
TARGET = "y"
SEED = 42


LOSS_CONFIGS = [
    {"name": "logistic_bce", "lr": 0.08, "epochs": 160, "l2": 0.0001},
    {"name": "linear_probability", "lr": 0.2, "epochs": 160, "l2": 0.0001},
    {"name": "squared_probability", "lr": 0.2, "epochs": 160, "l2": 0.0001},
    {"name": "quartic_probability", "lr": 0.5, "epochs": 160, "l2": 0.0001},
    {"name": "exponential_margin", "lr": 0.002, "epochs": 160, "l2": 0.0001},
    {"name": "squared_hinge_margin", "lr": 0.01, "epochs": 160, "l2": 0.0001},
    {"name": "cubic_hinge_margin", "lr": 0.002, "epochs": 160, "l2": 0.0001},
    {"name": "log_cosh_margin", "lr": 0.04, "epochs": 160, "l2": 0.0001},
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


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sample_weights(y):
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    pos_weight = neg / pos
    return np.where(y == 1, pos_weight, 1.0)


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
        raise ValueError(name)

    return loss, grad_z


def stratified_split(y, train_ratio=0.6, val_ratio=0.2, seed=SEED):
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


def prepare_data(drop_duration=False):
    df = pd.read_csv(DATA_PATH, sep=";")
    if drop_duration:
        df = df.drop(columns=["duration"])
    y = (df[TARGET] == "yes").astype(float).to_numpy()
    X_df = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False)
    X = X_df.to_numpy(dtype=float)

    train_idx, val_idx, test_idx = stratified_split(y)
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


def objective(X, y, w, loss_name, l2):
    losses, _ = loss_and_grad_z(loss_name, X @ w, y)
    weights = sample_weights(y)
    data_loss = np.sum(weights * losses) / np.sum(weights)
    reg_loss = l2 * np.sum(w[:-1] ** 2) / (2 * np.sum(weights))
    return float(data_loss + reg_loss)


def train(X_train, y_train, X_val, y_val, config):
    rng = np.random.default_rng(SEED)
    n, d = X_train.shape
    w = np.zeros(d)
    batch_size = 256
    train_losses = []
    val_losses = []
    start = time.perf_counter()

    for _ in range(config["epochs"]):
        idx = rng.permutation(n)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        for i in range(0, n, batch_size):
            Xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            weights = sample_weights(yb)
            _, grad_z = loss_and_grad_z(config["name"], Xb @ w, yb)
            grad_w = Xb.T @ (weights * grad_z) / np.sum(weights)
            grad_w[:-1] += config["l2"] * w[:-1] / np.sum(weights)
            grad_w = np.clip(grad_w, -50, 50)
            w -= config["lr"] * grad_w

        train_losses.append(objective(X_train, y_train, w, config["name"], config["l2"]))
        val_losses.append(objective(X_val, y_val, w, config["name"], config["l2"]))

    seconds = time.perf_counter() - start
    return w, train_losses, val_losses, seconds


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


def metrics(y_true, y_prob, threshold):
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


def run_setting(setting_name, drop_duration):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(drop_duration=drop_duration)
    rows = []
    curves = {}
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
            "threshold": threshold,
            "time_seconds": seconds,
            "final_train_objective": train_losses[-1],
            "final_val_objective": val_losses[-1],
        }
        for key, value in metrics(y_test, test_prob, threshold).items():
            row[f"test_{key}"] = value
        rows.append(row)
        curves[f"{setting_name}: {DISPLAY_NAMES[config['name']]}"] = train_losses
    return rows, curves


def save_loss_svg(loss_dict, path, title):
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
        f'<text x="500" y="34" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111"/>',
    ]
    for idx, (name, losses) in enumerate(loss_dict.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x:.2f},{y:.2f}" for i, value in enumerate(losses) for x, y in [xy(i, value, len(losses))])
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>')
        parts.append(f'<text x="{width - margin + 10}" y="{margin + 16 * idx}" font-size="10" font-family="Arial" fill="{color}">{name}</text>')
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
    with_rows, with_curves = run_setting("with_duration", drop_duration=False)
    without_rows, without_curves = run_setting("without_duration", drop_duration=True)
    rows = with_rows + without_rows

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "bank_custom_loss_comparison.csv", index=False)
    save_loss_svg(with_curves, OUTPUT_DIR / "bank_custom_loss_with_duration.svg", "Bank Marketing Custom Losses: With Duration")
    save_loss_svg(without_curves, OUTPUT_DIR / "bank_custom_loss_without_duration.svg", "Bank Marketing Custom Losses: Without Duration")

    columns = [
        "setting", "display_name", "threshold", "test_accuracy", "test_precision",
        "test_recall", "test_f1", "test_auc", "final_val_objective", "time_seconds",
    ]
    summary = [
        "# Bank Marketing Custom Loss Experiments",
        "",
        "## Results",
        "",
        markdown_table(rows, columns),
    ]
    (OUTPUT_DIR / "bank_custom_loss_results_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    run()
