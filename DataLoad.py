"""This File contains the Load the CSV of Indian Bank Transactions.
   """


import os
import pandas as pd
from exceptions import DataFileError


class DataLoader:
    """Reads the banking CSV.
       file_path must be a non-empty string else raise ValueError .
       if the column data is missing raises DataFileError.
    """

    REQUIRED_COLUMNS = ("transaction_amount", "transaction_direction",
                        "transaction_status")

    def __init__(self, file_path):
        """Store the file path"""

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        self.__file_path = file_path
        self.__data = None

    def load(self):
        """Read the CSV and return the number of rows loaded."""
        try:
            if not os.path.exists(self.__file_path):
                raise DataFileError(f"CSV file not found: {self.__file_path}")
            data = pd.read_csv(self.__file_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise DataFileError(f"CSV could not be read: {error}")
        else:
            missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
            if missing:
                raise DataFileError(f"Missing columns: {missing}")
            self.__data = data          # saved only when everything is fine
            return len(data)
        finally:
            print("Load attempt finished")

    def get_data(self):
        """Return a copy of the loaded DataFrame (DataFileError if load() not called)."""
        if self.__data is None:
            raise DataFileError("Call load() first")
        return self.__data.copy()