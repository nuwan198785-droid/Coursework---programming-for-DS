"""
    Here,in this StoreData python file, we implement the storage format classes.

    each format class provides its methods such as
        1.own write(), 
        2.read() 
        3.read_column(). 
        
    they all work the same way, the benchmark can loop through a list of StoreData objects
    and time each one without checking which format it is (which means polymorphism executed).

"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# create a lookup table (a Python dictionary) that matches each compression method to the extra ending added to the file name.
_Suffix = {None: "", "gzip": ".gz", "bz2": ".bz2", "zstd": ".zst", "snappy": ""}


class StorageFormat(ABC):
    """
        Base class for all storage formats. Each object represents one file format with one compression setting.

        Subclasses set their own NAME, EXTENSION and SUPPORTED values:

    """

    Name = "abstract"
    Extension = ""
    Supported = (None,)

    def __init__(self, compression=None):
        if compression not in self.Supported:
            raise ValueError(f"{self.Name} does not support compression {compression!r}; "
                             f"choose from {self.Supported}")
        self._compression = compression

    @property
    def compression(self):
        """str (or None) the codec in use."""
        return self._compression

    @property
    def label(self):
        """str: 'CSV '."""
        return f"{self.Name} ({self._compression or 'none'})"

    def path_in(self, folder):
        """
            Build and return the file path for this format inside the given folder.
            The file name includes the compression name, the format's file extension and the compression suffix.
        """
        tag = self._compression or "none"
        return Path(folder) / f"transactions_{tag}{self.Extension}{_Suffix.get(self._compression, '')}"

    @staticmethod
    def file_size_mb(path):
        """
            This method returns the size of a file in megabytes. 
            The benchmark can use it to compare how much disk space each format and compression setting takes
        """
        return os.path.getsize(path) / 1e6

    @abstractmethod
    def write(self, frame, path):
        """here , this method writes the whole DataFrame to path."""

    @abstractmethod
    def read(self, path):
        """in this method ,this read the whole file back into a DataFrame."""

    @abstractmethod
    def read_column(self, path, column):
        """Return one column as a Series ."""


class CSVFormat(StorageFormat):
    """
        in this class, it stores the data as a CSV file, with no compression, gzip or bz2.
        The CSV file format is a row-based text format that is easy to read and widely supported, 
    """
    Name, Extension, Supported = "CSV", ".csv", (None, "gzip", "bz2")

    def write(self, frame, path):
        """Save the DataFrame to a CSV file without the index column."""
        frame.to_csv(path, index=False, compression=self._compression)

    def read(self, path):
        """Load the whole CSV file into a DataFrame."""
        return pd.read_csv(path, compression=self._compression)

    def read_column(self, path, column):
        # In here, usecols keeps only the requested column, Eventhough when it come to the pandas,
        # it must still read through the whole file.
        return pd.read_csv(path, usecols=[column], compression=self._compression)[column]


class JSONFormat(StorageFormat):
    """
        in this class, it stores the data as a JSON Lines file, with no compression or gzip.
        also,each row is saved as one JSON record on its own line. The format is easy to read and share,
    """
    Name, Extension, Supported = "JSON", ".jsonl", (None, "gzip")

    def write(self, frame, path):
        """Save the DataFrame as JSON Lines, with one record per line."""
        frame.to_json(path, orient="records", lines=True, compression=self._compression)

    def read(self, path):
        """Load the whole JSON Lines file into a DataFrame."""
        return pd.read_json(path, orient="records", lines=True, compression=self._compression)

    def read_column(self, path, column):
        """Load the whole file, then return the requested column as a Series."""
        return self.read(path)[column]


class PickleFormat(StorageFormat):
    """
        in this class, it stores the data as a Python pickle file, with no compression, gzip or bz2.

        the python Pickle files saves the DataFrame as one binary object, so it usually reads 
        and writes quickly and keeps the data types unchanged. 
    """
    Name, Extension, Supported = "Pickle", ".pkl", (None, "gzip", "bz2")

    def write(self, frame, path):
        """Save the whole DataFrame to a pickle file."""
        frame.to_pickle(path, compression=self._compression)

    def read(self, path):
        """Load the whole pickle file back into a DataFrame."""
        return pd.read_pickle(path, compression=self._compression)

    def read_column(self, path, column):
        """Load the whole file, then return the requested column as a Series."""
        return self.read(path)[column]


class ParquetFormat(StorageFormat):
    """
        in this class,it stores the data as a Parquet file, with no compression, snappy, gzip or zstd.

        the Parquet file is a binary columnar format, so it keeps data types, 
        compresses well and can read one column without loading the rest of the file.
    """
    Name, Extension, Supported = "Parquet", ".parquet", (None, "snappy", "gzip", "zstd")

    def write(self, frame, path):
        """Save the DataFrame to a Parquet file without the index column."""
        frame.to_parquet(path, index=False, compression=self._compression)

    def read(self, path):
        """Load the whole Parquet file into a DataFrame."""
        return pd.read_parquet(path)

    def read_column(self, path, column):
        """Load only the requested column and return it as a Series."""
        return pd.read_parquet(path, columns=[column])[column]

    @staticmethod
    def available():
        """Return True if pyarrow is installed, otherwise False."""
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            return False


class YAMLFormat(StorageFormat):
    """
        in this class, it stores the data as a YAML list of records, with no compression.

        the YAML file format is easy for people to read, but it is very slow on large tables and the files are large,
    """
    Name, Extension, Supported = "YAML", ".yaml", (None,)

    def write(self, frame, path):
        """Convert the DataFrame to plain Python records and save them as YAML."""
        import yaml
        records = json.loads(frame.to_json(orient="records"))  # plain Python types
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(records, handle, sort_keys=False)

    def read(self, path):
        """Load the YAML file into a DataFrame, using the faster C loader if available."""
        import yaml
        with open(path, encoding="utf-8") as handle:
            loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            return pd.DataFrame(yaml.load(handle, Loader=loader))

    def read_column(self, path, column):
        """Load the whole file, then return the requested column as a Series."""
        return self.read(path)[column]
