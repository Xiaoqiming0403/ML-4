import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/bank_marketing/bank-additional/bank-additional-full.csv")
OUTPUT_DIR = Path("outputs/bank_marketing")
TARGET = "y"
SEED = 40


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sample_weights(y, pos_weight=1.0):
    return np.where(y == 1, pos_weight, 1.0)


def binary_cross_entropy(X, y, w, l2=0.0, pos_weight=1.0):
    p = np.clip(sigmoid(X @ w), 1e-15, 1 - 1e-15)
    weights = sample_weights(y, pos_weight)
    data_loss = -np.sum(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(weights)
    reg_loss = l2 * np.sum(w[:-1] * w[:-1]) / (2 * np.sum(weights))
    return data_loss + reg_loss


def gradient(X, y, w, l2=0.0, pos_weight=1.0):
    weights = sample_weights(y, pos_weight)
    error = weights * (sigmoid(X @ w) - y)
    g = X.T @ error / np.sum(weights)
    g[:-1] += l2 * w[:-1] / np.sum(weights)
    return g


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


def prepare_data(path=DATA_PATH, drop_duration=False):
    df = pd.read_csv(path, sep=";")
    if drop_duration and "duration" in df.columns:
        df = df.drop(columns=["duration"])

    y = (df[TARGET] == "yes").astype(float).to_numpy()
    X_df = df.drop(columns=[TARGET])
    X_df = pd.get_dummies(X_df, drop_first=False)
    feature_names = X_df.columns.to_numpy()
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

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train(
    X,
    y,
    batch_size,
    lr=0.1,
    epochs=80,
    l2=0.0,
    seed=SEED,
    use_scheduler=False,
    scheduler_factor=0.1,
    scheduler_patience=5,
    early_stop_patience=20,
    min_lr=1e-6,
    X_val=None,
    y_val=None,
    pos_weight=1.0,
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    losses = []
    updates = 0
    best_loss = math.inf
    bad_epochs = 0
    early_stop_bad_epochs = 0
    start = time.perf_counter()

    for _ in range(epochs):
        idx = rng.permutation(n)
        X_shuffled = X[idx]
        y_shuffled = y[idx]

        for i in range(0, n, batch_size):
            Xb = X_shuffled[i:i + batch_size]
            yb = y_shuffled[i:i + batch_size]
            w -= lr * gradient(Xb, yb, w, l2=l2, pos_weight=pos_weight)
            updates += 1

        epoch_loss = binary_cross_entropy(X, y, w, l2=l2, pos_weight=pos_weight)
        losses.append(epoch_loss)

        if use_scheduler:
            monitor_loss = epoch_loss
            if X_val is not None and y_val is not None:
                monitor_loss = binary_cross_entropy(X_val, y_val, w, l2=l2, pos_weight=pos_weight)

            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                bad_epochs = 0
                early_stop_bad_epochs = 0
            else:
                bad_epochs += 1
                early_stop_bad_epochs += 1
                if bad_epochs >= scheduler_patience:
                    lr *= scheduler_factor
                    bad_epochs = 0
                if lr <= min_lr or early_stop_bad_epochs >= early_stop_patience:
                    break

    seconds = time.perf_counter() - start
    return w, losses, seconds, updates


def predict_proba(X, w):
    return sigmoid(X @ w)


def predict(X, w, threshold=0.5):
    return (predict_proba(X, w) >= threshold).astype(int)


def metrics(y_true, y_pred, y_prob):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(y_true, y_score):
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return (np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold_by_f1(y_true, y_prob):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= threshold).astype(int)
        f1 = metrics(y_true, pred, y_prob)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate(name, w, losses, seconds, updates, X_train, y_train, X_val, y_val, X_test, y_test, threshold=0.5):
    row = {
        "experiment": name,
        "time_seconds": seconds,
        "updates": updates,
        "final_loss": losses[-1],
        "threshold": threshold,
    }
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        prob = predict_proba(X_split, w)
        pred = (prob >= threshold).astype(int)
        split_metrics = metrics(y_split, pred, prob)
        for key, value in split_metrics.items():
            row[f"{split_name}_{key}"] = value
    return row


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def save_loss_svg(loss_dict, path, title):
    width, height = 900, 520
    margin = 60
    all_losses = [v for losses in loss_dict.values() for v in losses]
    min_loss, max_loss = min(all_losses), max(all_losses)
    span = max(max_loss - min_loss, 1e-12)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def point(i, loss, n):
        x = margin + i * (width - 2 * margin) / max(n - 1, 1)
        y = height - margin - (loss - min_loss) * (height - 2 * margin) / span
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111"/>',
        f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="14" font-family="Arial">Epoch</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="14" font-family="Arial">Loss</text>',
    ]

    for idx, (name, losses) in enumerate(loss_dict.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{x:.2f},{y:.2f}" for i, loss in enumerate(losses) for x, y in [point(i, loss, len(losses))])
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        parts.append(f'<text x="{width - margin + 10}" y="{margin + idx * 22}" font-size="13" font-family="Arial" fill="{color}">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_confusion_svg(row, path, title):
    tn, fp, fn, tp = row["test_tn"], row["test_fp"], row["test_fn"], row["test_tp"]
    cells = [[tn, fp], [fn, tp]]
    max_value = max(tn, fp, fn, tp)
    labels = [["TN", "FP"], ["FN", "TP"]]
    width, height = 520, 420
    cell = 130
    x0, y0 = 150, 100

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>',
        f'<text x="{x0 + cell}" y="80" text-anchor="middle" font-size="14" font-family="Arial">Predicted</text>',
        f'<text x="45" y="{y0 + cell}" transform="rotate(-90 45 {y0 + cell})" text-anchor="middle" font-size="14" font-family="Arial">Actual</text>',
        f'<text x="{x0 + cell * 0.5}" y="96" text-anchor="middle" font-size="13" font-family="Arial">No</text>',
        f'<text x="{x0 + cell * 1.5}" y="96" text-anchor="middle" font-size="13" font-family="Arial">Yes</text>',
        f'<text x="{x0 - 18}" y="{y0 + cell * 0.5 + 5}" text-anchor="end" font-size="13" font-family="Arial">No</text>',
        f'<text x="{x0 - 18}" y="{y0 + cell * 1.5 + 5}" text-anchor="end" font-size="13" font-family="Arial">Yes</text>',
    ]

    for r in range(2):
        for c in range(2):
            value = cells[r][c]
            intensity = int(235 - 165 * value / max_value)
            fill = f"rgb({intensity},{intensity + 12},{255})"
            x = x0 + c * cell
            y = y0 + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#111"/>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + 55}" text-anchor="middle" font-size="18" font-family="Arial">{labels[r][c]}</text>')
            parts.append(f'<text x="{x + cell / 2}" y="{y + 88}" text-anchor="middle" font-size="24" font-family="Arial" font-weight="bold">{value}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.6f}" if "time" not in col else f"{value:.3f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + body)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data()
    class_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))

    dataset_summary = {
        "rows": len(y_train) + len(y_val) + len(y_test),
        "features_after_one_hot": X_train.shape[1] - 1,
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),
        "positive_rate": float(np.mean(np.concatenate([y_train, y_val, y_test]))),
    }

    gd_configs = [
        ("Full-batch GD", len(X_train), 0.1, 80, 0.0, False, 1.0, False),
        ("Mini-batch GD", 256, 0.1, 80, 0.0, False, 1.0, False),
        ("SGD", 1, 0.01, 30, 0.0, False, 1.0, False),
        ("Mini-batch + threshold tuning", 256, 0.1, 80, 0.0, False, 1.0, True),
        ("Mini-batch + weight decay + scheduler", 256, 0.1, 300, 0.0001, True, 1.0, True),
        ("Mini-batch + class weight + scheduler", 256, 0.05, 300, 0.0001, True, class_pos_weight, True),
    ]

    gd_rows = []
    gd_losses = {}
    best_row = None
    best_w = None
    for name, batch_size, lr, epochs, l2, use_scheduler, pos_weight, tune_threshold in gd_configs:
        w, losses, seconds, updates = train(
            X_train,
            y_train,
            batch_size,
            lr=lr,
            epochs=epochs,
            l2=l2,
            use_scheduler=use_scheduler,
            X_val=X_val,
            y_val=y_val,
            pos_weight=pos_weight,
        )
        threshold = 0.5
        if tune_threshold:
            threshold, _ = best_threshold_by_f1(y_val, predict_proba(X_val, w))
        row = evaluate(name, w, losses, seconds, updates, X_train, y_train, X_val, y_val, X_test, y_test, threshold=threshold)
        row.update({
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": len(losses),
            "max_epochs": epochs,
            "l2": l2,
            "scheduler": use_scheduler,
            "pos_weight": pos_weight,
        })
        gd_rows.append(row)
        gd_losses[name] = losses
        if best_row is None or row["val_f1"] > best_row["val_f1"]:
            best_row = row
            best_w = w

    lr_rows = []
    lr_losses = {}
    for lr in [0.02, 0.05, 0.1, 0.2]:
        name = f"lr={lr}"
        w, losses, seconds, updates = train(X_train, y_train, 256, lr=lr, epochs=80, l2=0.0)
        row = evaluate(name, w, losses, seconds, updates, X_train, y_train, X_val, y_val, X_test, y_test)
        row.update({"batch_size": 256, "learning_rate": lr, "epochs": 80, "l2": 0.0})
        lr_rows.append(row)
        lr_losses[name] = losses

    l2_rows = []
    l2_losses = {}
    for l2 in [0.0, 0.1, 1.0, 10.0]:
        name = f"L2={l2}"
        w, losses, seconds, updates = train(X_train, y_train, 256, lr=0.1, epochs=80, l2=l2)
        row = evaluate(name, w, losses, seconds, updates, X_train, y_train, X_val, y_val, X_test, y_test)
        row.update({"batch_size": 256, "learning_rate": 0.1, "epochs": 80, "l2": l2})
        l2_rows.append(row)
        l2_losses[name] = losses

    duration_rows = []
    for drop_duration in [False, True]:
        name = "with duration" if not drop_duration else "without duration"
        d_X_train, d_y_train, d_X_val, d_y_val, d_X_test, d_y_test, _ = prepare_data(drop_duration=drop_duration)
        d_pos_weight = float(np.sum(d_y_train == 0) / np.sum(d_y_train == 1))
        w, losses, seconds, updates = train(
            d_X_train,
            d_y_train,
            256,
            lr=0.05,
            epochs=300,
            l2=0.0001,
            use_scheduler=True,
            X_val=d_X_val,
            y_val=d_y_val,
            pos_weight=d_pos_weight,
        )
        threshold, _ = best_threshold_by_f1(d_y_val, predict_proba(d_X_val, w))
        row = evaluate(name, w, losses, seconds, updates, d_X_train, d_y_train, d_X_val, d_y_val, d_X_test, d_y_test, threshold=threshold)
        row.update({
            "drop_duration": drop_duration,
            "batch_size": 256,
            "learning_rate": 0.05,
            "epochs": len(losses),
            "max_epochs": 300,
            "l2": 0.0001,
            "scheduler": True,
            "pos_weight": d_pos_weight,
        })
        duration_rows.append(row)

    save_csv(gd_rows, OUTPUT_DIR / "gradient_descent_comparison.csv")
    save_csv(lr_rows, OUTPUT_DIR / "learning_rate_comparison.csv")
    save_csv(l2_rows, OUTPUT_DIR / "l2_comparison.csv")
    save_csv(duration_rows, OUTPUT_DIR / "duration_comparison.csv")
    save_loss_svg(gd_losses, OUTPUT_DIR / "gd_loss_curves.svg", "Gradient Descent Loss Curves")
    save_loss_svg(lr_losses, OUTPUT_DIR / "learning_rate_loss_curves.svg", "Learning Rate Loss Curves")
    save_loss_svg(l2_losses, OUTPUT_DIR / "l2_loss_curves.svg", "L2 Regularization Loss Curves")
    save_confusion_svg(best_row, OUTPUT_DIR / "best_confusion_matrix.svg", f"Best Model Confusion Matrix: {best_row['experiment']}")

    prob = predict_proba(X_test, best_w)
    pred = (prob >= 0.5).astype(int)
    pd.DataFrame({"y_true": y_test.astype(int), "y_prob": prob, "y_pred": pred}).to_csv(OUTPUT_DIR / "best_test_predictions.csv", index=False)

    summary_lines = [
        "# Bank Marketing Logistic Regression Results",
        "",
        "## Dataset",
        "",
        f"- Rows: {dataset_summary['rows']}",
        f"- Train/Validation/Test: {dataset_summary['train_rows']} / {dataset_summary['val_rows']} / {dataset_summary['test_rows']}",
        f"- Features after one-hot encoding: {dataset_summary['features_after_one_hot']}",
        f"- Positive class rate: {dataset_summary['positive_rate']:.6f}",
        "",
        "## Gradient Descent Comparison",
        "",
        markdown_table(gd_rows, ["experiment", "batch_size", "epochs", "threshold", "pos_weight", "updates", "time_seconds", "train_accuracy", "val_accuracy", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc", "final_loss"]),
        "",
        "## Learning Rate Comparison",
        "",
        markdown_table(lr_rows, ["experiment", "learning_rate", "time_seconds", "val_f1", "test_accuracy", "test_f1", "test_auc", "final_loss"]),
        "",
        "## L2 Regularization Comparison",
        "",
        markdown_table(l2_rows, ["experiment", "l2", "time_seconds", "val_f1", "test_accuracy", "test_f1", "test_auc", "final_loss"]),
        "",
        "## Duration Feature Comparison",
        "",
        markdown_table(duration_rows, ["experiment", "epochs", "threshold", "pos_weight", "time_seconds", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc", "final_loss"]),
        "",
        "## Best Model",
        "",
        f"- Selected by validation F1: {best_row['experiment']}",
        f"- Test accuracy: {best_row['test_accuracy']:.6f}",
        f"- Test precision: {best_row['test_precision']:.6f}",
        f"- Test recall: {best_row['test_recall']:.6f}",
        f"- Test F1: {best_row['test_f1']:.6f}",
        f"- Test AUC: {best_row['test_auc']:.6f}",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    run()
