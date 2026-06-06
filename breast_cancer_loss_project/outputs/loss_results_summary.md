# Breast Cancer Custom Loss Function Experiments

## Results

| setting | display_name | train_size | threshold | test_accuracy | test_precision | test_recall | test_f1 | test_auc | final_train_objective | time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_train | Logistic BCE | 397 | 0.540000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.997796 | 0.050972 | 0.195919 |
| full_train | Linear probability loss | 397 | 0.160000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.023125 | 0.175351 |
| full_train | Squared probability loss | 397 | 0.400000 | 0.988636 | 1.000000 | 0.969697 | 0.984615 | 1.000000 | 0.007682 | 0.154679 |
| full_train | Quartic probability loss | 397 | 0.510000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 1.000000 | 0.001323 | 0.194751 |
| full_train | Exponential margin loss | 397 | 0.560000 | 0.965909 | 1.000000 | 0.909091 | 0.952381 | 0.998898 | 0.141724 | 0.173848 |
| full_train | Squared hinge margin loss | 397 | 0.550000 | 0.954545 | 1.000000 | 0.878788 | 0.935484 | 0.996694 | 0.055568 | 0.160183 |
| full_train | Cubic hinge margin loss | 397 | 0.510000 | 0.965909 | 1.000000 | 0.909091 | 0.952381 | 0.997796 | 0.070906 | 0.161018 |
| full_train | Log-cosh margin loss | 397 | 0.480000 | 0.943182 | 1.000000 | 0.848485 | 0.918033 | 0.996694 | 0.091684 | 0.176759 |
| small_train_80 | Logistic BCE | 79 | 0.320000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.994490 | 0.025100 | 0.063350 |
| small_train_80 | Linear probability loss | 79 | 0.270000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.996694 | 0.018515 | 0.057350 |
| small_train_80 | Squared probability loss | 79 | 0.390000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.995041 | 0.004780 | 0.048937 |
| small_train_80 | Quartic probability loss | 79 | 0.440000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.995041 | 0.000705 | 0.055771 |
| small_train_80 | Exponential margin loss | 79 | 0.360000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.151324 | 0.054962 |
| small_train_80 | Squared hinge margin loss | 79 | 0.520000 | 0.943182 | 1.000000 | 0.848485 | 0.918033 | 0.991185 | 0.013129 | 0.050073 |
| small_train_80 | Cubic hinge margin loss | 79 | 0.460000 | 0.977273 | 1.000000 | 0.939394 | 0.968750 | 0.996694 | 0.038152 | 0.053107 |
| small_train_80 | Log-cosh margin loss | 79 | 0.470000 | 0.943182 | 1.000000 | 0.848485 | 0.918033 | 0.972452 | 0.063136 | 0.058235 |