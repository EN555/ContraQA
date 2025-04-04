import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap


def bootstrap_mean(data: np.array, alpha: float) -> tuple[float, float, float]:
    ci_res = bootstrap(
        (data,), np.nanmean, vectorized=True, confidence_level=1 - alpha, rng=42, n_resamples=10_000
    )
    f1_value = np.nanmean(data)
    ci = max(f1_value - ci_res.confidence_interval.low, ci_res.confidence_interval.high - f1_value)
    return f1_value, ci, ci_res


def paired_test_mean(
    scores_a,
    scores_b,
    model_name_a,
    model_name_b,
    paired_p_value: float = 0.01,
    unpaired_alpha=0.05,
) -> str:  # 0.01 or 0.05 is pretty common
    res_gt = scipy.stats.wilcoxon(
        scores_a,
        scores_b,
        zero_method="pratt",
        alternative="greater",
        nan_policy="omit",
    )
    res_lt = scipy.stats.wilcoxon(
        scores_b,
        scores_a,
        zero_method="pratt",
        alternative="greater",
        nan_policy="omit",
    )
    mean_a, ci_a, _ = bootstrap_mean(scores_a, alpha=unpaired_alpha)
    mean_b, ci_b, _ = bootstrap_mean(scores_b, alpha=unpaired_alpha)

    assert not pd.isna(res_gt.pvalue)
    assert not pd.isna(res_lt.pvalue)

    is_significant_gt = res_gt.pvalue < paired_p_value
    is_significant_lt = res_lt.pvalue < paired_p_value
    if is_significant_gt:
        prefix = "GT-SIGNIFICANT"
        assert not is_significant_lt
        res = res_gt
    elif is_significant_lt:
        prefix = "LT-SIGNIFICANT"
        assert not is_significant_gt
        res = res_lt
    else:
        prefix = "NOT SIGNIFICANT"
        res = res_gt

    print(
        f"--- {prefix} {model_name_a}(f1={mean_a:.2f}±{ci_a:.1f}) ≥ {model_name_b}(f1={mean_b:.2f}±{ci_b:.1f}) - {res.pvalue.round(7)} ---"
    )
    return prefix


if __name__ == "__main__":
    arr = np.arange(10, 1010)
    arr2 = np.arange(1000)
    print(paired_test_mean(arr, arr2, "model1", "model2"))
    print(paired_test_mean(arr2, arr, "model1", "model2"))
