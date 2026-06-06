# Bank Marketing Custom Loss Experiments

## Results

| setting | display_name | threshold | test_accuracy | test_precision | test_recall | test_f1 | test_auc | final_val_objective | time_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_duration | Logistic BCE | 0.710000 | 0.901202 | 0.545600 | 0.734914 | 0.626263 | 0.936083 | 0.331544 | 1.733018 |
| with_duration | Linear probability loss | 0.950000 | 0.875713 | 0.471360 | 0.851293 | 0.606759 | 0.931807 | 0.124892 | 1.629014 |
| with_duration | Squared probability loss | 0.790000 | 0.895983 | 0.525969 | 0.774784 | 0.626580 | 0.933325 | 0.046169 | 1.522946 |
| with_duration | Quartic probability loss | 0.570000 | 0.899866 | 0.540329 | 0.743534 | 0.625850 | 0.936009 | 0.007729 | 1.650905 |
| with_duration | Exponential margin loss | 0.570000 | 0.891734 | 0.515228 | 0.656250 | 0.577251 | 0.918911 | 0.679820 | 1.572421 |
| with_duration | Squared hinge margin loss | 0.590000 | 0.902901 | 0.555556 | 0.689655 | 0.615385 | 0.935564 | 0.431879 | 1.410833 |
| with_duration | Cubic hinge margin loss | 0.540000 | 0.898653 | 0.537836 | 0.712284 | 0.612888 | 0.934423 | 0.497744 | 1.501612 |
| with_duration | Log-cosh margin loss | 0.540000 | 0.884695 | 0.492350 | 0.762931 | 0.598478 | 0.931806 | 0.196925 | 1.653151 |
| without_duration | Logistic BCE | 0.610000 | 0.866974 | 0.433014 | 0.585129 | 0.497709 | 0.793168 | 0.548125 | 1.545739 |
| without_duration | Linear probability loss | 0.950000 | 0.866489 | 0.431090 | 0.579741 | 0.494485 | 0.782433 | 0.266536 | 1.498543 |
| without_duration | Squared probability loss | 0.630000 | 0.869280 | 0.438277 | 0.570043 | 0.495550 | 0.793416 | 0.092117 | 1.467476 |
| without_duration | Quartic probability loss | 0.540000 | 0.868795 | 0.437653 | 0.578664 | 0.498376 | 0.792121 | 0.012539 | 1.728311 |
| without_duration | Exponential margin loss | 0.590000 | 0.874014 | 0.450091 | 0.534483 | 0.488670 | 0.794640 | 0.846021 | 1.568986 |
| without_duration | Squared hinge margin loss | 0.550000 | 0.866003 | 0.430490 | 0.587284 | 0.496809 | 0.793155 | 0.742618 | 1.447073 |
| without_duration | Cubic hinge margin loss | 0.530000 | 0.866974 | 0.432584 | 0.580819 | 0.495860 | 0.794653 | 0.783580 | 1.552586 |
| without_duration | Log-cosh margin loss | 0.590000 | 0.871101 | 0.441536 | 0.545259 | 0.487946 | 0.792028 | 0.301431 | 1.642635 |