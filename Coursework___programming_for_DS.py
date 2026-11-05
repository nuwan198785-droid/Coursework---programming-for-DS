

"""Main Project File"""

"""
project scope is to find which transaction characteristics are associated with fraud, 
and how much better than random can a simple, explainable rule find it?
"""

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt       
import numpy as np                    
import pandas as pd                   

from DataLoad import DataCleaner, DataLoader,Get_Hex_Digest #DataCleaner,DataLoader,Get_Hex_Digest  - DataLoad,FraudAnalyzer - Analysis,
from Analysis import FraudAnalyzer
from exceptions import BankingDataError  

Root = Path(__file__).resolve().parent # Project folder (the folder containing this file),
Raw_Data = Root / "data" / "raw" / "IndianBankingTransactions.csv"   # Original dataset. This file should not be changed.
Checksum = Raw_Data.with_suffix(".csv.sha256") # SHA-256 checksum file
Processed = Root / "data" / "processed" # Folder for cleaned data.
OUT = Root / "outputs" # Folder for result files.
FIG = Root / "figures" # Folder for charts and figures.
CUTOFF = "2023-01-01"   # development data before it, test data from it

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e" # Colours used in all charts.

# create default chart style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def pct(x, digits=2):
    """Convert a decimal to percentage text"""
    return f"{100 * x:.{digits}f}%"


def save(fig, name):
    """Save a chart to the figures folder"""
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved figures/{name}")
    

"""
def rule_performance(df, threshold=200_000):
    
        This function tests a simple fraud rule. this give a flag any transaction of at least 200,000. 
        It then checks how well the rule works in two periods:
        
            1.Development: transactions before the cutoff year
            2.Test: transactions from the cutoff year onwards
    Using two periods shows whether the rule still works on later data, rather than only on the data used to create it.
    
    
    cutoff_year = int(CUTOFF[:4])
    periods = {"development": df[df["year"] < cutoff_year], "test": df[df["year"] >= cutoff_year]}
    results = {}
    for period, part in periods.items():
        flagged = part["transaction_amount"] >= threshold
        fraud = part["is_fraud"] == 1
        amount = part["transaction_amount"]
        precision = fraud[flagged].mean()
        base_rate = fraud.mean()
        results[period] = {
            "flagged_share": flagged.mean(),
            "recall": (flagged & fraud).sum() / fraud.sum(),
            "fraud_value_captured": amount[flagged & fraud].sum() / amount[fraud].sum(),
            "precision": precision,
            "base_rate": base_rate,
            "lift": precision / base_rate,
        }
    return pd.DataFrame.from_dict(results, orient="index")
    """

# 1. OBTAIN
def obtain(args):
    """
            this function contains the 1st step in OSEMN (obtain): verify and load the raw dataset.

            On the first run, the SHA-256 checksum of the raw file is calculated and saved. 
            On every run, the file is checked against that checksum and loaded.
    """
    print("\n[1] OBTAIN")
    if not Checksum.exists():
        digest = Get_Hex_Digest(Raw_Data)
        Checksum.write_text(f"{digest}  {Raw_Data.name}\n")
        print(f"    first run: recorded SHA-256 {digest[:16]}... in {Checksum.name}")
    expected = Checksum.read_text().split()[0]

    loader = DataLoader(Raw_Data, Expected_sha256_Val=expected)
    raw = loader.load()                              # verifies checksum + schema
    meta = {
        "file": Raw_Data.name, "sha256": expected, "size_bytes": Raw_Data.stat().st_size,
        "rows": len(raw), "columns": raw.shape[1],
        "source_url": args.source_url, "licence": args.licence,
        "extracted_on": args.extracted_on, "pipeline_run_on": date.today().isoformat(),
        "method": "Manual download of the CSV from the source page; stored unchanged in "
                  "data/raw/ and verified by SHA-256 on every run.",
        "load_seconds": round(loader.last_load_seconds, 2),
    }
    (OUT / "obtain_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"    {len(raw):,} rows x {raw.shape[1]} columns, checksum verified, "
          f"loaded in {meta['load_seconds']} s")
    return raw


# 2. SCRUB
def scrub(raw):
    """
        this function contains the 2nd step in OSEMN (Scrub): clean the raw dataset.

        This function is the Scrub step of your OSEMN pipeline. 
        It cleans the raw data, records every cleaning action, saves the results, 
        prints a summary, and returns the clean data for the next step.
    """
    print("\n[2] SCRUB")
    cleaner = DataCleaner(raw)
    clean = cleaner.run()
    log = cleaner.log.to_frame()
    log.to_csv(OUT / "scrub_log.csv", index=False)
    clean.to_pickle(Processed / "clean_transactions.pkl")
    width = log["issue"].str.len().max()
    for row in log.itertuples():
        print(f"    {row.step:>2}. {row.issue:<{width}}  {row.n_affected:>8,}  -> {row.action}")
    print(f"    rows in: {len(raw):,}   rows out: {len(clean):,}   (no rows deleted)")
    return clean


#  3. EXPLORE
def explore(df, analyzer):
    """
        this function contains the 3rd step in OSEMN  (Explore): describe the data and find what is linked to fraud.

        in this function Saves summary statistics and fraud rates to the outputs folder, 
        and creates five charts in the figures folder:
            1. Figure 1 - distribution of transaction amounts
            2. Figure 2 - fraud rate by amount decile
            3. Figure 3 - fraud rate by transaction type
            4. Figure 4 - fraud rate by transaction type and amount band
            5. Figure 5 - factors with little effect on the fraud rate

        Returns the DataFrame 
    """
    
    print("\n[3] EXPLORE")
    numeric = ["transaction_amount", "account_balance", "credit_score", "emi_amount"]
    summary = df[numeric].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).T
    summary["skew"] = df[numeric].skew()
    summary.round(2).to_csv(OUT / "summary_statistics.csv")
    print(summary[["mean", "50%", "99%", "max", "skew"]].round(1).to_string())

    base = analyzer.base_rate
    print(f"    overall fraud rate {pct(base)} ({int(df['is_fraud'].sum()):,} of {len(df):,})")

    # Figure 1 - amount distribution on a log scale
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(df["transaction_amount"].min()), 7, 70)
    ax.hist(df["transaction_amount"], bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    med, mean = df["transaction_amount"].median(), df["transaction_amount"].mean()
    for value, label, colour in [(med, "median", INK), (mean, "mean", ORANGE)]:
        ax.axvline(value, color=colour, linestyle="--", linewidth=1.2)
        ax.text(value * 1.12, ax.get_ylim()[1] * 0.92, f"{label}\nRs {value:,.0f}",
                color=colour, fontsize=8, va="top")
    ax.set_title("Transaction amounts are heavily right-skewed")
    ax.set_xlabel("Transaction amount (Rs, log scale)")
    ax.set_ylabel("Transactions")
    save(fig, "fig1_amount_distribution.png")

    # Figure 2 - fraud rate by amount decile
    df = df.assign(amount_decile=pd.qcut(df["transaction_amount"], 10, labels=False) + 1)
    dec = analyzer.__class__(df).fraud_rate_by("amount_decile").sort_index()
    edges = df.groupby("amount_decile")["transaction_amount"].max()
    dec.assign(upper_amount=edges).to_csv(OUT / "fraud_by_amount_decile.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = [ORANGE if d == 10 else BLUE for d in dec.index]
    ax.bar(dec.index, dec["rate"] * 100, color=colours, width=0.7)
    ax.errorbar(dec.index, dec["rate"] * 100,
                yerr=[(dec["rate"] - dec["ci_low"]) * 100, (dec["ci_high"] - dec["rate"]) * 100],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(base * 100, color=GREY, linestyle=":", linewidth=1)
    ax.text(0.6, base * 100 + 0.05, f"overall {pct(base)}", color=MUTED, fontsize=8)
    ax.set_xticks(dec.index)
    def short(v):
        return f"{v / 1000:,.1f}k" if v >= 1000 else f"{v:,.0f}"
    ax.set_xticklabels([f"D{d}\nup to {short(edges[d])}" if d < 10 else "D10\nabove 51.3k"
                        for d in dec.index], fontsize=7)
    ax.set_xlabel("Amount decile (Rs)")
    ax.set_title("Only the top 10% of amounts carries a higher fraud rate")
    ax.set_ylabel("Fraud rate (%) with 95% CI")
    save(fig, "fig2_fraud_by_amount_decile.png")

    # Figure 3 - fraud rate by transaction type
    by_type = analyzer.fraud_rate_by("transaction_type").sort_values("rate")
    by_type.to_csv(OUT / "fraud_by_transaction_type.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(by_type))
    colours = [ORANGE if t in ("RTGS", "Cheque") else BLUE for t in by_type.index]
    ax.barh(y, by_type["rate"] * 100, color=colours, height=0.65)
    ax.errorbar(by_type["rate"] * 100, y,
                xerr=[(by_type["rate"] - by_type["ci_low"]) * 100,
                      (by_type["ci_high"] - by_type["rate"]) * 100],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axvline(base * 100, color=GREY, linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  (n={n:,})" for t, n in zip(by_type.index, by_type["n"])])
    ax.set_title("RTGS and cheques stand out; other types overlap the average")
    ax.set_xlabel("Fraud rate (%) with 95% CI   (dotted line = overall rate)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_fraud_by_transaction_type.png")

    # Figure 4 -  (type x amount) band heat map
    band = pd.cut(df["transaction_amount"], [0, 5_000, 50_000, 200_000, np.inf], right=False,
                  labels=["< Rs 5k", "Rs 5k-50k", "Rs 50k-2L", ">= Rs 2L"])
    cross = df.assign(band=band).groupby(["transaction_type", "band"], observed=True)["is_fraud"] \
              .agg(["mean", "size"])
    rate = cross["mean"].unstack()
    count = cross["size"].unstack()
    rate = rate.where(count >= 300)                      
    rate = rate.loc[by_type.index[::-1]]
    count = count.loc[rate.index]
    cross.to_csv(OUT / "fraud_by_type_and_amount_band.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(rate * 100, cmap="Blues", aspect="auto", vmin=0, vmax=4)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            value = rate.iat[i, j]
            if np.isnan(value):
                ax.text(j, i, "n<300", ha="center", va="center", color=GREY, fontsize=7)
            else:
                ax.text(j, i, f"{value * 100:.2f}%\nn={int(count.iat[i, j]):,}", ha="center",
                        va="center", fontsize=7, color="white" if value > 0.022 else INK)
    ax.set_xticks(range(rate.shape[1]))
    ax.set_xticklabels(rate.columns)
    ax.set_yticks(range(rate.shape[0]))
    ax.set_yticklabels(rate.index)
    ax.grid(False)
    ax.set_title("Whatever the type, fraud jumps only at Rs 2 lakh and over")
    fig.colorbar(im, ax=ax, label="Fraud rate (%)", shrink=0.8)
    save(fig, "fig4_type_by_amount_heatmap.png")

    # Figure 5 - factors with little effect on the fraud rate
    df["hour_band"] = pd.cut(df["transaction_hour"], [-1, 5, 11, 17, 23],
                             labels=["00-05", "06-11", "12-17", "18-23"])
    df["credit_band"] = pd.cut(df["credit_score"], [299, 500, 650, 750, 900],
                               labels=["300-500", "501-650", "651-750", "751-899"])
    flat = ["channel", "kyc_status", "hour_band", "credit_band"]
    titles = ["Channel", "KYC status", "Hour of day", "Credit score"]
    a2 = analyzer.__class__(df)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True)
    for ax, column, title in zip(axes, flat, titles):
        t = a2.fraud_rate_by(column).sort_index()
        x = np.arange(len(t))
        ax.bar(x, t["rate"] * 100, color=BLUE, width=0.65)
        ax.errorbar(x, t["rate"] * 100, yerr=[(t["rate"] - t["ci_low"]) * 100,
                                              (t["ci_high"] - t["rate"]) * 100],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.axhline(base * 100, color=GREY, linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in t.index], rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.4)
    axes[0].set_ylabel("Fraud rate (%) with 95% CI")
    fig.suptitle("Customer and context fields barely move the fraud rate", fontweight="bold")
    save(fig, "fig5_flat_factors.png")

    # Yearly stability 
    yearly = df[~df["dq_outside_window"]].groupby("year")["is_fraud"].agg(["size", "mean"])
    yearly.to_csv(OUT / "fraud_by_year.csv")
    print("    fraud rate by year:", ", ".join(f"{y}: {pct(r)}" for y, r in yearly["mean"].items()))
    return df


#  4. MODEL
def model(df):
    """
        this function contains the 4th step in OSEMN (Model): test, segment and predict fraud.

        in this class, it gives answers for three questions: 
            1.which fields are linked to fraud
            2.which amount groups carry the most risk
            3.how well fraud can be predicted on later data.
        
        To acquire results, this function devided in to 4 parts
            1. Statistical comparison
            2. Segmentation
            3. Predictive check
            4. Compares the model
            
        here we have to execute chi-square tests to find which fields are linked to fraud, groups
        transactions into amount tiers, and trains a model on data before 2023 and tests it on 2023 onward. 
        The model is compared with the simple rule of amount >= Rs 2 lakh and with random review in Figure 6.
        
    """

    print("\n[4] MODEL")
    analyzer = FraudAnalyzer(df)

    # 4a. Statistical comparison: which fields are associated with fraud at all
    columns = ["transaction_type", "amount_decile", "channel", "kyc_status", "account_type",
               "transaction_status", "merchant_category", "state", "hour_band", "credit_band",
               "dq_exceeds_balance", "dq_rtgs_below_min"]
    tests = pd.DataFrame([analyzer.chi_square(c) for c in columns]) \
              .sort_values("cramers_v", ascending=False)
    tests.to_csv(OUT / "chi_square_tests.csv", index=False)
    print(tests[["column", "chi2", "p_value", "cramers_v"]].round(4).to_string(index=False))

    # 4b. Segmentation into amount tiers: first all data, then the development period only
    tiers = analyzer.risk_tiers()
    tiers.to_csv(OUT / "risk_tiers.csv")
    print(tiers[["n", "frauds", "rate", "share_of_rows", "share_of_frauds",
                 "share_of_fraud_value"]].round(4).to_string())
    
    development = df[df["year"] < int(CUTOFF[:4])]
    development_tiers = FraudAnalyzer(development).risk_tiers()
    development_tiers.to_csv(OUT / "risk_tiers_development.csv")

    # Performance of the simple rule on the development and test periods
    development = df[df["timestamp"] < pd.Timestamp(CUTOFF)]
    development_tiers = FraudAnalyzer(development).risk_tiers()
    development_tiers.to_csv(OUT / "risk_tiers_development.csv")


    # 4c. Evaluate the review rule on development data and on later test data
    rule = analyzer.evaluate_rule(cutoff=CUTOFF)
    rule.to_csv(OUT / "rule_evaluation.csv", index_label="period")
    test = rule.loc["test"]
    print(f"    rule: review every transaction of Rs 2 lakh or more | split at {CUTOFF}")
    print(f"    development: {int(rule.loc['development', 'n']):,} rows, "
          f"{int(rule.loc['development', 'frauds']):,} frauds | "
          f"test: {int(test['n']):,} rows, {int(test['frauds']):,} frauds")
    print(f"    test: flags {pct(test['flagged_share'], 1)} -> {pct(test['recall'], 1)} of frauds, "
          f"{pct(test['fraud_value_captured'], 1)} of fraud value")
    print(f"    test precision {pct(test['precision'])} "
          f"(95% CI {pct(test['precision_ci_low'])} to {pct(test['precision_ci_high'])}), "
          f"{test['lift']:.1f} times the {pct(test['base_rate'])} fraud rate")



    # Figure 6 - simple rule vs random review
    labels = ["Development\n(before 2023)", "Test\n(2023 onward)"]
    x = np.arange(len(labels))
    width = 0.24
    series = [("flagged_share", "Reviewed (random review catches this share)", GREY),
              ("recall", "Fraud cases caught", BLUE),
              ("fraud_value_captured", "Fraud value caught", ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, (column, label, colour) in zip((-width, 0, width), series):
        values = rule[column].to_numpy() * 100
        bars = ax.bar(x + offset, values, width=width * 0.92, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=2, fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share of period total (%)")
    ax.set_title("The Rs 2 lakh rule finds fraud value better than fraud cases")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    save(fig, "fig6_rule_vs_random.png")
    return tests, tiers, development_tiers, rule


# 5. INTERPRET
def interpret(df, tests, tiers, development_tiers, rule): 
    """
        this function contains the 5th step in OSEMN (Interpret): turn the results into a written summary.

        in here, this class combines the overall fraud rate, the strongest chi-square result, 
        the Tier A figures for all data and the development period, and the performance of the Rs 2 lakh rule on the test data. 
        The rule is also compared with random review. 
    """
    
    print("\n[5] INTERPRET")
    top = tiers.iloc[0]
    development_top = development_tiers.iloc[0]
    test = rule.loc["test"]
    lines = [
        "# Results summary (generated)",
        "",
        f"- Overall fraud rate: {pct(df['is_fraud'].mean())} "
        f"({int(df['is_fraud'].sum()):,} of {len(df):,}).",
        f"- Strongest association (Cramer's V): {tests.iloc[0]['column']} "
        f"= {tests.iloc[0]['cramers_v']:.3f}; all others below "
        f"{tests.iloc[1]['cramers_v']:.3f}. Values under 0.1 are conventionally 'negligible'.",
        f"- Tier A (Rs 2 lakh and over), all data: {pct(top['share_of_rows'])} of transactions, "
        f"fraud rate {pct(top['rate'])}, {pct(top['share_of_frauds'], 1)} of fraud cases, "
        f"{pct(top['share_of_fraud_value'], 1)} of fraud value.",
        f"- Tier A, development period: {pct(development_top['share_of_rows'])} of transactions, "
        f"fraud rate {pct(development_top['rate'])}, "
        f"{pct(development_top['share_of_frauds'], 1)} of fraud cases, "
        f"{pct(development_top['share_of_fraud_value'], 1)} of fraud value.",
        f"- Rule 'amount >= Rs 2 lakh' on test data reviews {pct(test['flagged_share'], 1)} and "
        f"catches {pct(test['recall'], 1)} of cases, {pct(test['fraud_value_captured'], 1)} of "
        f"value. Random review of the same share would catch about "
        f"{pct(test['flagged_share'], 1)} of both.",
        f"- Rule precision on test data: {pct(test['precision'])}, {test['lift']:.1f} times the "
        f"test fraud rate of {pct(test['base_rate'])}.",
    ]
    (OUT / "results_summary.md").write_text("\n".join(lines) + "\n")
    for line in lines[2:]:
        print("   ", line)



def main(argv=None):
    """
        This is the main function, here we run the full OSEMN pipeline from start to finish.
        The program stops with a clear message if the raw file is missing or fails validation.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-url", default="FILL IN: URL of the page you downloaded from")
    parser.add_argument("--licence", default="FILL IN: licence shown on the source page")
    parser.add_argument("--extracted-on", default="FILL IN: YYYY-MM-DD you downloaded it")
    args = parser.parse_args(argv)
    for folder in (Processed, OUT, FIG):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}")
    try:
        raw = obtain(args)
    except FileNotFoundError:
        sys.exit(f"Put the raw file at {Raw_Data} first.")
    except BankingDataError as exc:
        sys.exit(f"Stopped: {exc}")
    clean = scrub(raw)
    analyzer = FraudAnalyzer(clean)
    explored = explore(clean, analyzer)
    tests, tiers, development_tiers, rule = model(explored)
    interpret(explored, tests, tiers, development_tiers, rule)
    print("\nDone. Tables in outputs/, figures in figures/.")


if __name__ == "__main__":
    main()
