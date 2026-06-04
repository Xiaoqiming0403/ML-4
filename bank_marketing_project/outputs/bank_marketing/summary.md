# Bank Marketing Logistic Regression Results

## Dataset

- Rows: 41188
- Train/Validation/Test: 24712 / 8237 / 8239
- Features after one-hot encoding: 63
- Positive class rate: 0.112654

## Gradient Descent Comparison

| experiment | batch_size | epochs | threshold | pos_weight | updates | time_seconds | train_accuracy | val_accuracy | test_accuracy | test_precision | test_recall | test_f1 | test_auc | final_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full-batch GD | 24712 | 80 | 0.500000 | 1.000000 | 80 | 0.494 | 0.905552 | 0.907612 | 0.904721 | 0.633147 | 0.366379 | 0.464164 | 0.933199 | 0.276060 |
| Mini-batch GD | 256 | 80 | 0.500000 | 1.000000 | 7760 | 0.601 | 0.911946 | 0.908947 | 0.908848 | 0.645799 | 0.422414 | 0.510749 | 0.933888 | 0.207462 |
| SGD | 1 | 30 | 0.500000 | 1.000000 | 741360 | 12.371 | 0.910246 | 0.902877 | 0.906542 | 0.619697 | 0.440733 | 0.515113 | 0.927151 | 0.218624 |
| Mini-batch + threshold tuning | 256 | 80 | 0.260000 | 1.000000 | 7760 | 0.543 | 0.907413 | 0.900328 | 0.902901 | 0.555846 | 0.686422 | 0.614272 | 0.933888 | 0.207462 |
| Mini-batch + weight decay + scheduler | 256 | 32 | 0.260000 | 1.000000 | 3104 | 0.233 | 0.908142 | 0.901056 | 0.904600 | 0.562832 | 0.685345 | 0.618076 | 0.932480 | 0.207968 |
| Mini-batch + class weight + scheduler | 256 | 33 | 0.670000 | 7.876437 | 3201 | 0.242 | 0.895557 | 0.891830 | 0.895254 | 0.523500 | 0.780172 | 0.626569 | 0.937029 | 0.327344 |

## Learning Rate Comparison

| experiment | learning_rate | time_seconds | val_f1 | test_accuracy | test_f1 | test_auc | final_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr=0.02 | 0.020000 | 0.567 | 0.509596 | 0.909576 | 0.512753 | 0.932654 | 0.207992 |
| lr=0.05 | 0.050000 | 0.567 | 0.507937 | 0.909091 | 0.512687 | 0.933026 | 0.207715 |
| lr=0.1 | 0.100000 | 0.548 | 0.503311 | 0.908848 | 0.510749 | 0.933888 | 0.207462 |
| lr=0.2 | 0.200000 | 0.544 | 0.508229 | 0.908848 | 0.511386 | 0.934811 | 0.207463 |

## L2 Regularization Comparison

| experiment | l2 | time_seconds | val_f1 | test_accuracy | test_f1 | test_auc | final_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.0 | 0.000000 | 0.547 | 0.503311 | 0.908848 | 0.510749 | 0.933888 | 0.207462 |
| L2=0.1 | 0.100000 | 0.542 | 0.502653 | 0.909334 | 0.512084 | 0.933715 | 0.207560 |
| L2=1.0 | 1.000000 | 0.541 | 0.497640 | 0.908727 | 0.498667 | 0.932887 | 0.208274 |
| L2=10.0 | 10.000000 | 0.543 | 0.442308 | 0.906057 | 0.418919 | 0.930183 | 0.217575 |

## Duration Feature Comparison

| experiment | epochs | threshold | pos_weight | time_seconds | test_accuracy | test_precision | test_recall | test_f1 | test_auc | final_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with duration | 33 | 0.670000 | 7.876437 | 0.242 | 0.895254 | 0.523500 | 0.780172 | 0.626569 | 0.937029 | 0.327344 |
| without duration | 23 | 0.700000 | 7.876437 | 0.170 | 0.873043 | 0.443269 | 0.496767 | 0.468496 | 0.787223 | 0.534856 |

## Best Model

- Selected by validation F1: Mini-batch + class weight + scheduler
- Test accuracy: 0.895254
- Test precision: 0.523500
- Test recall: 0.780172
- Test F1: 0.626569
- Test AUC: 0.937029