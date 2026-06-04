import numpy as np
import pandas as pd


TRAIN_PATH = r"C:\Users\Young\OneDrive\Desktop\课程资料\人工智能的编程基础\train.csv"
TEST_PATH = r"C:\Users\Young\OneDrive\Desktop\课程资料\人工智能的编程基础\test.csv"
ANSWER_PATH = r"C:\Users\Young\OneDrive\Desktop\课程资料\人工智能的编程基础\sample_submission.csv"
PRED_PATH = "test_predictions.csv"

TARGET = "NObeyesdad"
ID_COL = "id"

alpha = 0.4
beta = 0.5


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def loss(X, Y, W):
    prob = np.clip(softmax(X @ W), 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(Y * np.log(prob), axis=1))


def gradient(X, Y, W):
    return X.T @ (softmax(X @ W) - Y) / len(Y)


def stratified_split(y, train_ratio=0.8, seed=40):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        split = int(len(idx) * train_ratio)
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])

    train_idx = rng.permutation(np.array(train_idx))
    val_idx = rng.permutation(np.array(val_idx))
    return train_idx, val_idx


def prepare_data(train_df, test_df):
    y_text = train_df[TARGET].to_numpy()
    classes = np.array(sorted(train_df[TARGET].unique()))
    class_to_id = {name: i for i, name in enumerate(classes)}
    y = np.array([class_to_id[name] for name in y_text])

    train_features = train_df.drop(columns=[TARGET])
    test_features = test_df.copy()
    test_ids = test_features[ID_COL].to_numpy()

    all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    all_features = all_features.drop(columns=[ID_COL])
    all_features = pd.get_dummies(all_features, drop_first=False)

    X_all = all_features.to_numpy(dtype=float)
    X_train_all = X_all[:len(train_df)]
    X_test_final = X_all[len(train_df):]

    train_idx, val_idx = stratified_split(y)
    X_train = X_train_all[train_idx]
    y_train = y[train_idx]
    X_val = X_train_all[val_idx]
    y_val = y[val_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test_final = (X_test_final - mean) / std

    X_train = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_val = np.column_stack([X_val, np.ones(X_val.shape[0])])
    X_test_final = np.column_stack([X_test_final, np.ones(X_test_final.shape[0])])

    return X_train, y_train, X_val, y_val, X_test_final, test_ids, classes


def train(X, y, lr=0.2, epochs=350, batch_size=256, seed=40):
    n, d = X.shape
    num_classes = len(np.unique(y))
    rng = np.random.default_rng(seed)
    W = np.zeros((d, num_classes))
    Y = one_hot(y, num_classes)
    losses = []

    for epoch in range(epochs):
        idx = rng.permutation(n)
        X_shuffled = X[idx]
        Y_shuffled = Y[idx]

        for i in range(0, n, batch_size):
            Xb = X_shuffled[i:i + batch_size]
            Yb = Y_shuffled[i:i + batch_size]
            g = gradient(Xb, Yb, W)
            if loss(Xb, Yb, W - lr * g) > loss(Xb, Yb, W) - alpha * lr * np.sum(g * g):
                lr = lr * beta
            W = W - lr * g

        losses.append(loss(X, Y, W))
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d}, loss={losses[-1]:.6f}, lr={lr:.6g}")

    return W, losses


def predict(X, W):
    return np.argmax(softmax(X @ W), axis=1)


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train, y_train, X_val, y_val, X_test_final, test_ids, classes = prepare_data(train_df, test_df)

W, losses = train(X_train, y_train)

y_train_pred = predict(X_train, W)
y_val_pred = predict(X_val, W)

train_accuracy = accuracy(y_train_pred, y_train)
val_accuracy = accuracy(y_val_pred, y_val)

print(f"Train Accuracy: {train_accuracy:.6f}")
print(f"Validation Accuracy: {val_accuracy:.6f}")

test_pred = predict(X_test_final, W)
test_labels = classes[test_pred]
submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_labels})
submission.to_csv(PRED_PATH, index=False)
print(f"Saved test predictions to {PRED_PATH}")

answer = pd.read_csv(ANSWER_PATH)
score_df = submission.merge(answer, on=ID_COL, suffixes=("_pred", "_true"))
test_accuracy = np.mean(score_df[f"{TARGET}_pred"] == score_df[f"{TARGET}_true"])
print(f"Test Accuracy: {test_accuracy:.6f}")
