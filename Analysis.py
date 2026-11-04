"""

This file provide thw process of the Fraud analysis on the scrubbed dataset.
basically, in here, Analyze fraud patterns in the cleaned transaction dataset.

This module provides the FraudAnalyzer class to identify transaction
characteristics associated with fraud and measure how well a simple,
explainable model detects fraud compared with random selection.

Main methods:
    1. fraud_rate_by(): Calculate segment fraud rates, 95% Wilson confidence
                        intervals, and lift.
    2.chi_square() : Test independence and measure effect size with Cramer's V.
    3.risk_tiers(): Compare amount-based tiers by their share of rows, fraud
                    cases, and fraud value.
    4.evaluate_rule(): Evaluate an amount-based review rule on development
                       and later test periods.

"""

import numpy as np
import pandas as pd
from scipy import stats

from BankTransaction import High_val_Threshold

Mid_Val_Threshold = 50_000.0


def wilson_interval(successes, trials, z=1.96):
    """
        this function calculates the Wilson score confidence interval for a proportion. 
        which means Return (low, high) Wilson score intervals for proportions.
    
        This gives the likely range of a true rate, such as a fraud rate.
        It is more reliable than the normal approximation when the rate is close to 0, as with fraud rates of about 0.9 %.

    """


    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need trials > 0 and 0 <= successes <= trials")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


class FraudAnalyzer:
    """
        in this class , fraud patterns are analyzed with banking transaction data.

        This class uses the is_fraud column to compare fraud across different groups of transactions and to evaluate a fraud detection rule.
    """

    _Req_Data = ("is_fraud", "transaction_amount", "timestamp", "transaction_type")

    def __init__(self, frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        missing = [c for c in self._Req_Data if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing columns: {missing}")
        if not frame["is_fraud"].isin([0, 1]).all():
            raise ValueError("is_fraud must contain only 0 and 1")
        self._df = frame
        self._base_rate = float(frame["is_fraud"].mean())

    @property
    def base_rate(self):
        """overall fraud rate."""
        return self._base_rate

    def _require_column(self, column):
        if not isinstance(column, str) or column not in self._df.columns:
            raise ValueError(f"unknown column {column!r}")

    #  comparisons
    def fraud_rate_by(self, column, min_count=1):
        """
            in this function fraud rate is calculated for each group in a column.

            Groups the transactions by the given column, calculates fraud
            statistics for each group and sorts the groups from the highest to
            the lowest fraud rate.
        """
        self._require_column(column)
        grouped = self._df.groupby(column, observed=True)["is_fraud"].agg(["size", "sum"])
        grouped.columns = ["n", "frauds"]
        grouped = grouped[grouped["n"] >= min_count]
        grouped["rate"] = grouped["frauds"] / grouped["n"]
        grouped["ci_low"], grouped["ci_high"] = wilson_interval(grouped["frauds"], grouped["n"])
        grouped["lift"] = grouped["rate"] / self._base_rate
        return grouped.sort_values("rate", ascending=False)

    def chi_square(self, column):
        """
            this function is implemented to test whether a column is associated with fraud.

            this runs a chi-square test of independence between the given column and is_fraud, then calculates Cramer's V 
            to measure the strength of the association.
        """
        self._require_column(column)
        table = pd.crosstab(self._df[column], self._df["is_fraud"])
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        k = min(table.shape) - 1
        return {"column": column, "chi2": float(chi2), "dof": int(dof),
                "p_value": float(p_value), "cramers_v": float(np.sqrt(chi2 / (n * k)))}

    def risk_tiers(self):
        """
        This function splits transactions into three risk tiers based on their amount. 
        It then shows how much of the transaction volume, fraud and fraud value falls into each tier.
        """
        df = self._df
        tier = pd.cut(df["transaction_amount"],
                      bins=[0, Mid_Val_Threshold, High_val_Threshold, np.inf],
                      labels=["C: under Rs 50k", "B: Rs 50k - 2 lakh", "A: Rs 2 lakh and over"],
                      right=False)
        fraud_value = df["transaction_amount"].where(df["is_fraud"] == 1, 0.0)
        out = pd.DataFrame({
            "n": df.groupby(tier, observed=True).size(),
            "frauds": df.groupby(tier, observed=True)["is_fraud"].sum(),
            "fraud_value": fraud_value.groupby(tier, observed=True).sum(),
        })
        out["rate"] = out["frauds"] / out["n"]
        out["ci_low"], out["ci_high"] = wilson_interval(out["frauds"], out["n"])
        out["share_of_rows"] = out["n"] / out["n"].sum()
        out["share_of_frauds"] = out["frauds"] / out["frauds"].sum()
        out["share_of_fraud_value"] = out["fraud_value"] / out["fraud_value"].sum()
        return out.iloc[::-1]

   
    def evaluate_rule(self, cutoff="2023-01-01", threshold=High_val_Threshold):
        """
            this function evaluates the transaction amount rule for each time period.

            The rule flags transactions at or above the threshold. For each period, this calculates how accurately the rule identifies fraud,
            how much fraud it catches and how much fraud value it captures.

        """
        df = self._df
        cutoff = pd.Timestamp(cutoff)
        periods = {
            "development": df["timestamp"] < cutoff,
            "test": df["timestamp"] >= cutoff,
        }
        results = {}

        for period, mask in periods.items():
            part = df[mask]
            flagged = part["transaction_amount"] >= threshold
            total_frauds = int(part["is_fraud"].sum())
            caught_frauds = int(part.loc[flagged, "is_fraud"].sum())
            n_flagged = int(flagged.sum())

            if part.empty or total_frauds == 0 or n_flagged == 0:
                raise ValueError(f"{period} period cannot be evaluated")

            base_rate = total_frauds / len(part)
            precision = caught_frauds / n_flagged
            ci_low, ci_high = wilson_interval(caught_frauds, n_flagged)
            fraud_value = part.loc[part["is_fraud"] == 1, "transaction_amount"]
            caught_value = part.loc[
                flagged & (part["is_fraud"] == 1), "transaction_amount"
            ]

            results[period] = {
                "n": len(part),
                "frauds": total_frauds,
                "base_rate": base_rate,
                "flagged_share": n_flagged / len(part),
                "precision": precision,
                "precision_ci_low": float(ci_low),
                "precision_ci_high": float(ci_high),
                "recall": caught_frauds / total_frauds,
                "fraud_value_captured": caught_value.sum() / fraud_value.sum(),
                "lift": precision / base_rate,
            }

        return pd.DataFrame(results).T
