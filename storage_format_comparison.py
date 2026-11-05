"""
we have to do fair comparison of  CSV, JSON, pickle, and Parquet.

here we measure
    1. Write Time
    2. Read Time
    3. File Size
    4. Memory use after loading
    5. Data type preservation


"""
import argparse
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd

from StoreData import CSVFormat, JSONFormat, ParquetFormat, PickleFormat, YAMLFormat

Root = Path(__file__).resolve().parent
CLEAN = Root / "data" / "processed" / "clean_transactions.pkl"
OUT, FIG = Root / "outputs", Root / "figures"
COLUMN = "transaction_amount"
YAML_ROWS = 50_000   # YAML is time consuming,therefore we took on a sample because the full table takes far too long


def measure_median_runtime(function, repeats):
    """
        Run a function several times and measure its typical runtime.
        Return the median of repeated calls.
    """
    times, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - started)
    return statistics.median(times), result


def count_preserved_dtypes(original, restored):
    """
        This function counts how many columns keep the same pandas data type after the data is saved and loaded again. 
        This process is called a round trip.
    """
    return sum(str(original[c].dtype) == str(restored[c].dtype)
               for c in original.columns if c in restored.columns)


def benchmark_storage_format(storage_format, frame, folder, repeats):
    """
        Measure the performance of one storage format.
        from this, it tests one storage format by writing the data, reading it back, and reading a single column. 
        It then records the results, deletes the test file, prints a summary line, and returns the results as a dictionary.
    """
    path = storage_format.path_in(folder)
    write_s, _ = measure_median_runtime(lambda: storage_format.write(frame, path), repeats)
    read_s, restored = measure_median_runtime(lambda: storage_format.read(path), repeats)
    column_s, _ = measure_median_runtime(lambda: storage_format.read_column(path, COLUMN), repeats)
    row = {"format": storage_format.Name, "compression": storage_format.compression or "none", "rows": len(frame),
           "size_mb": storage_format.file_size_mb(path), "write_s": write_s, "read_s": read_s,
           "read_column_s": column_s, "dtypes_kept": count_preserved_dtypes(frame, restored),
           "columns": frame.shape[1]}
    path.unlink()
    print(f"  {storage_format.label:<20} {row['size_mb']:>8.1f} MB  write {write_s:>7.2f}s  "
          f"read {read_s:>7.2f}s  column {column_s:>7.2f}s  "
          f"dtypes {row['dtypes_kept']}/{row['columns']}", flush=True)
    return row


def chart(results):
    """
        This function creates a scatter chart that compares each storage format’s file size with its full read time, 
        Each point represents one format and compression option. 
        Points in the lower left are best because those files are smaller and faster to read.
    
    """
    full = results[results["format"] != "YAML"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for row in full.itertuples():
        colour = "#2a78d6" if row.format == "Parquet" else "#8a8984"
        ax.scatter(row.size_mb, row.read_s, s=36, color=colour, zorder=3)
        ax.annotate(f"{row.format} {row.compression}", (row.size_mb, row.read_s),
                    xytext=(0, 7), textcoords="offset points", fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(15, 700)
    ax.set_ylim(0.08, 30)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([20, 50, 100, 200, 500])
    ax.set_yticks([0.1, 1, 10])
    ax.set_xlabel("File size (MB, log scale)")
    ax.set_ylabel("Full read time (s, log scale)")
    ax.set_title("Lower left is better: small files that read quickly", fontweight="bold")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_storage_benchmark.png", dpi=200)


def main():
    """
        Run the storage benchmark or redraw its chart from saved results.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chart-only", action="store_true", help="redraw the chart from saved results")
    args = parser.parse_args()
    if args.chart_only:
        chart(pd.read_csv(OUT / "storage_benchmark.csv"))
        return
    frame = pd.read_pickle(CLEAN)
    print(f"Python {platform.python_version()} | pandas {pd.__version__} | "
          f"{len(frame):,} rows x {frame.shape[1]} columns | {args.repeats} repeats (median)")
    formats = [CSVFormat(c) for c in (None, "gzip", "bz2")]
    formats += [JSONFormat(c) for c in (None, "gzip")]
    formats += [PickleFormat(c) for c in (None, "gzip", "bz2")]
    if ParquetFormat.available():
        formats += [ParquetFormat(c) for c in (None, "snappy", "gzip", "zstd")]
    else:
        print("  pyarrow is not installed: Parquet rows skipped")
    folder = Path(tempfile.mkdtemp(prefix="storage_"))
    try:
        rows = [benchmark_storage_format(storage_format, frame, folder, args.repeats) for storage_format in formats]
        rows.append(benchmark_storage_format(YAMLFormat(), frame.sample(YAML_ROWS, random_state=42), folder, 1))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    results = pd.DataFrame(rows)
    results.round(3).to_csv(OUT / "storage_benchmark.csv", index=False)
    chart(results)


if __name__ == "__main__":
    main()
