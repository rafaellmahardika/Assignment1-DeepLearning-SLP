import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
n_epochs = 5
initial_weight = 0.5
threshold = 0.5

TRAIN_DATA = np.array([
    [1, 5.1, 3.5, 1.4, 0.2, 0],
    [2, 4.9, 3, 1.4, 0.2, 0],
    [3, 4.7, 3.2, 1.3, 0.2, 0],
    [4, 4.6, 3.1, 1.5, 0.2, 0],
    [5, 5, 3.6, 1.4, 0.2, 0],
    [6, 5.4, 3.9, 1.7, 0.4, 0],
    [7, 4.6, 3.4, 1.4, 0.3, 0],
    [8, 5, 3.4, 1.5, 0.2, 0],
    [9, 4.4, 2.9, 1.4, 0.2, 0],
    [10, 4.9, 3.1, 1.5, 0.1, 0],
    [11, 5.4, 3.7, 1.5, 0.2, 0],
    [12, 4.8, 3.4, 1.6, 0.2, 0],
    [13, 4.8, 3, 1.4, 0.1, 0],
    [14, 4.3, 3, 1.1, 0.1, 0],
    [15, 5.8, 4, 1.2, 0.2, 0],
    [16, 5.7, 4.4, 1.5, 0.4, 0],
    [17, 5.4, 3.9, 1.3, 0.4, 0],
    [18, 5.1, 3.5, 1.4, 0.3, 0],
    [19, 5.7, 3.8, 1.7, 0.3, 0],
    [20, 5.1, 3.8, 1.5, 0.3, 0],
    [21, 5.4, 3.4, 1.7, 0.2, 0],
    [22, 5.1, 3.7, 1.5, 0.4, 0],
    [23, 4.6, 3.6, 1, 0.2, 0],
    [24, 5.1, 3.3, 1.7, 0.5, 0],
    [25, 4.8, 3.4, 1.9, 0.2, 0],
    [26, 5, 3, 1.6, 0.2, 0],
    [27, 5, 3.4, 1.6, 0.4, 0],
    [28, 5.2, 3.5, 1.5, 0.2, 0],
    [29, 5.2, 3.4, 1.4, 0.2, 0],
    [30, 4.7, 3.2, 1.6, 0.2, 0],
    [31, 4.8, 3.1, 1.6, 0.2, 0],
    [32, 5.4, 3.4, 1.5, 0.4, 0],
    [33, 5.2, 4.1, 1.5, 0.1, 0],
    [34, 5.5, 4.2, 1.4, 0.2, 0],
    [35, 4.9, 3.1, 1.5, 0.1, 0],
    [36, 5, 3.2, 1.2, 0.2, 0],
    [37, 5.5, 3.5, 1.3, 0.2, 0],
    [38, 4.9, 3.1, 1.5, 0.1, 0],
    [39, 4.4, 3, 1.3, 0.2, 0],
    [40, 5.1, 3.4, 1.5, 0.2, 0],

    [41, 7, 3.2, 4.7, 1.4, 1],
    [42, 6.4, 3.2, 4.5, 1.5, 1],
    [43, 6.9, 3.1, 4.9, 1.5, 1],
    [44, 5.5, 2.3, 4, 1.3, 1],
    [45, 6.5, 2.8, 4.6, 1.5, 1],
    [46, 5.7, 2.8, 4.5, 1.3, 1],
    [47, 6.3, 3.3, 4.7, 1.6, 1],
    [48, 4.9, 2.4, 3.3, 1, 1],
    [49, 6.6, 2.9, 4.6, 1.3, 1],
    [50, 5.2, 2.7, 3.9, 1.4, 1],
    [51, 5, 2, 3.5, 1, 1],
    [52, 5.9, 3, 4.2, 1.5, 1],
    [53, 6, 2.2, 4, 1, 1],
    [54, 6.1, 2.9, 4.7, 1.4, 1],
    [55, 5.6, 2.9, 3.6, 1.3, 1],
    [56, 6.7, 3.1, 4.4, 1.4, 1],
    [57, 5.6, 3, 4.5, 1.5, 1],
    [58, 5.8, 2.7, 4.1, 1, 1],
    [59, 6.2, 2.2, 4.5, 1.5, 1],
    [60, 5.6, 2.5, 3.9, 1.1, 1],
    [61, 5.9, 3.2, 4.8, 1.8, 1],
    [62, 6.1, 2.8, 4, 1.3, 1],
    [63, 6.3, 2.5, 4.9, 1.5, 1],
    [64, 6.1, 2.8, 4.7, 1.2, 1],
    [65, 6.4, 2.9, 4.3, 1.3, 1],
    [66, 6.6, 3, 4.4, 1.4, 1],
    [67, 6.8, 2.8, 4.8, 1.4, 1],
    [68, 6.7, 3, 5, 1.7, 1],
    [69, 6, 2.9, 4.5, 1.5, 1],
    [70, 5.7, 2.6, 3.5, 1, 1],
    [71, 5.5, 2.4, 3.8, 1.1, 1],
    [72, 5.5, 2.4, 3.7, 1, 1],
    [73, 5.8, 2.7, 3.9, 1.2, 1],
    [74, 6, 2.7, 5.1, 1.6, 1],
    [75, 5.4, 3, 4.5, 1.5, 1],
    [76, 6, 3.4, 4.5, 1.6, 1],
    [77, 6.7, 3.1, 4.7, 1.5, 1],
    [78, 6.3, 2.3, 4.4, 1.3, 1],
    [79, 5.6, 3, 4.1, 1.3, 1],
    [80, 5.5, 2.5, 4, 1.3, 1]
], dtype=float)


VAL_DATA = np.array([
    [1, 5, 3.5, 1.3, 0.3, 0],
    [2, 4.5, 2.3, 1.3, 0.3, 0],
    [3, 4.4, 3.2, 1.3, 0.2, 0],
    [4, 5, 3.5, 1.6, 0.6, 0],
    [5, 5.1, 3.8, 1.9, 0.4, 0],
    [6, 4.8, 3, 1.4, 0.3, 0],
    [7, 5.1, 3.8, 1.6, 0.2, 0],
    [8, 4.6, 3.2, 1.4, 0.2, 0],
    [9, 5.3, 3.7, 1.5, 0.2, 0],
    [10, 5, 3.3, 1.4, 0.2, 0],

    [11, 5.5, 2.6, 4.4, 1.2, 1],
    [12, 6.1, 3, 4.6, 1.4, 1],
    [13, 5.8, 2.6, 4, 1.2, 1],
    [14, 5, 2.3, 3.3, 1, 1],
    [15, 5.6, 2.7, 4.2, 1.3, 1],
    [16, 5.7, 3, 4.2, 1.2, 1],
    [17, 5.7, 2.9, 4.2, 1.3, 1],
    [18, 6.2, 2.9, 4.3, 1.3, 1],
    [19, 5.1, 2.5, 3, 1.1, 1],
    [20, 5.7, 2.8, 4.1, 1.3, 1]
], dtype=float)


X_train = TRAIN_DATA[:, 1:5]
y_train = TRAIN_DATA[:, 5].astype(int)

X_val = VAL_DATA[:, 1:5]
y_val = VAL_DATA[:, 5].astype(int)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


weights = np.full(5, initial_weight, dtype=float)

train_records = []
val_records = []


for epoch in range(1, n_epochs + 1):

    for i, (x, target) in enumerate(zip(X_train, y_train)):

        x_with_bias = np.r_[1.0, x]

        z = float(np.dot(weights, x_with_bias))
        pred_prob = float(sigmoid(z))
        pred = 1 if pred_prob > threshold else 0

        error = pred_prob - target
        squared_error = error ** 2

        train_records.append([
            epoch,
            i + 1,
            *x,
            int(target),
            *weights,
            z,
            pred_prob,
            pred,
            error,
            squared_error
        ])

        gradient = (
            2
            * (pred_prob - target)
            * (1 - pred_prob)
            * pred_prob
        )

        if i < len(X_train) - 1:
            weights = weights - learning_rate * gradient * x_with_bias


    for i, (x, target) in enumerate(zip(X_val, y_val)):

        x_with_bias = np.r_[1.0, x]

        z = float(np.dot(weights, x_with_bias))
        pred_prob = float(sigmoid(z))
        pred = 1 if pred_prob > threshold else 0

        error = pred_prob - target
        squared_error = error ** 2

        val_records.append([
            epoch,
            i + 1,
            *x,
            int(target),
            *weights,
            z,
            pred_prob,
            pred,
            error,
            squared_error
        ])


def calculate_metrics(records):

    results = []

    for epoch in range(1, n_epochs + 1):

        rows = [r for r in records if r[0] == epoch]

        actual = np.array(
            [r[6] for r in rows],
            dtype=int
        )

        prediction = np.array(
            [r[14] for r in rows],
            dtype=int
        )

        squared_errors = np.array(
            [r[16] for r in rows],
            dtype=float
        )

        tp = np.sum((actual == 1) & (prediction == 1))
        tn = np.sum((actual == 0) & (prediction == 0))
        fp = np.sum((actual == 0) & (prediction == 1))
        fn = np.sum((actual == 1) & (prediction == 0))

        accuracy = (tp + tn) / len(rows)

        sse = np.sum(squared_errors)
        mse = sse / len(rows)

        results.append({
            "epoch": epoch,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "accuracy": accuracy,
            "SSE": sse,
            "MSE": mse
        })

    return results


train_metrics = calculate_metrics(train_records)
val_metrics = calculate_metrics(val_records)


print("\nEpoch | Train Acc | Val Acc | Train MSE | Val MSE")
print("-" * 55)

for train, val in zip(train_metrics, val_metrics):

    print(
        f"{train['epoch']:5d} | "
        f"{train['accuracy'] * 100:9.2f}% | "
        f"{val['accuracy'] * 100:7.2f}% | "
        f"{train['MSE']:9.6f} | "
        f"{val['MSE']:7.6f}"
    )


print("\nTP, TN, FP, FN")

for train, val in zip(train_metrics, val_metrics):

    print(
        f"Epoch {train['epoch']}: "
        f"Train(TP={train['TP']}, TN={train['TN']}, "
        f"FP={train['FP']}, FN={train['FN']}) | "
        f"Val(TP={val['TP']}, TN={val['TN']}, "
        f"FP={val['FP']}, FN={val['FN']})"
    )


epochs = list(range(1, n_epochs + 1))

train_accuracy = np.array([
    x["accuracy"] for x in train_metrics
]) * 100

val_accuracy = np.array([
    x["accuracy"] for x in val_metrics
]) * 100

train_mse = np.array([
    x["MSE"] for x in train_metrics
])

val_mse = np.array([
    x["MSE"] for x in val_metrics
])


plt.figure(figsize=(7, 4.5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.xticks(epochs)
plt.ylim(0, 105)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


plt.figure(figsize=(7, 4.5))

plt.plot(
    epochs,
    train_mse,
    marker="o",
    label="Training"
)

plt.plot(
    epochs,
    val_mse,
    marker="o",
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")

plt.xticks(epochs)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()