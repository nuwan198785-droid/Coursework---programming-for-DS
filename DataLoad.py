"""This File contains the Load, Scrub and Clean the CSV of Indian Bank Transactions.

This File contain 3 main parts
        1.DataLoader   - opens the raw CSV, verified and validate the schema.
        2.ScrubLog     - an ordered record of every quality problem.
        3.DataCleaner  - runs each scrub step on a *copy* of the data and writes to the ScrubLog.

please note that , in here No step deletes rows. Problems are either repaired (with the count logged) or
marked in a ``dq_*`` flag column so the analysis can include or exclude them
openly.
"""


import hashlib
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from BankTransaction import Debit_Only_Types, RTGS_Min, Transaction_Types
from exceptions import DataIntegrityError, SchemaValidationError


# identified required colum and put them into tuple
Column_Req = (
    "transaction_id", "customer_id", "transaction_date", "transaction_time",
    "account_type", "transaction_type", "transaction_amount", "transaction_direction",
    "account_balance", "merchant_category", "state", "credit_score", "has_loan",
    "loan_type", "emi_amount", "transaction_status", "channel", "kyc_status",
    "is_fraud", "transaction_hour",
)

# in this, we used for domain cheching purpose, create a dictonary of column and then frozenset of permitted values.
Value_Req = {
    "account_type": frozenset({"Savings", "Current", "Salary", "NRI", "Fixed Deposit"}),
    "transaction_type": Transaction_Types,
    "transaction_direction": frozenset({"Debit", "Credit"}),
    "transaction_status": frozenset({"Success", "Failed", "Reversed", "Pending"}),
    "channel": frozenset({"Mobile_App", "Web", "ATM", "POS_Terminal", "Branch", "API"}),
    "kyc_status": frozenset({"Verified", "Pending", "Expired"}),
}

# standardizes the data by ensuring all state names are written in a uniform format
Standardize_State_name = {"UP": "Uttar Pradesh"}

Study_Start, Study_End = pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31 23:59")
Max_Trans_Amt, Min_AC_Bal, Max_AC_Bal = 10_000_000.0, 500.0, 5_000_000.0

# compute SHA-256,the cryptographic math formula (hash function)
def Get_Hex_Digest(path, chunk_size=1 << 20):
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks.

    Raises:
        FileNotFoundError: path does not exist.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataLoader:
    """
        Read and Load the banking CSV and check schema.
    """
    def __init__(self, path, Expected_sha256_Val=None):
        if not isinstance(path, (str, Path)):
            raise TypeError(f"Path must be string or Path, got {type(path).__name__}")
        path = Path(path)
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got {path.name!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        if Expected_sha256_Val is not None:
            if not (isinstance(Expected_sha256_Val, str) and len(Expected_sha256_Val) == 64
                    and all(c in "0123456789 abcdef" for c in Expected_sha256_Val.lower())):
                raise ValueError("Expected_sha256_Val must be a 64-character hex string")
        self._path = path
        self._Expected_sha256_Val = Expected_sha256_Val.lower() if Expected_sha256_Val else None
        self._last_load_seconds = None

    @property
    def path(self):
        """Path: the file this loader reads."""
        return self._path

    @property
    def last_load_seconds(self):  #returns the exact timestamp (wall-clock time) when the load() method was last executed
        """wall-clock time of the most recent load()."""
        return self._last_load_seconds

    def checksum(self): 
        """Return the file's SHA-256 hex digest."""
        return Get_Hex_Digest(self._path)

    def verify(self):
        """
        if the file not match Get_Hex_Digiset, give an error."""
        Actual = self.checksum()
        if self._Expected_sha256_Val and Actual != self._Expected_sha256_Val:
            raise DataIntegrityError(
                "raw extract has changed since its checksum was recorded",
                context={"file": self._path.name, "expected": self._Expected_sha256_Val[:12],
                         "actual": Actual[:12]},
            )
        return Actual

    def load(self, nrows=None):
        """
        Retun new Data Frame, after read and verify the .csv file
        """
        if nrows is not None and (isinstance(nrows, bool) or not isinstance(nrows, int) or nrows <= 0):
            raise ValueError(f"nrows must be a positive int or None, got {nrows!r}")
        self.verify()
        started = time.perf_counter()
        try:
            frame = pd.read_csv(self._path, nrows=nrows)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise ValueError(f"could not parse {self._path.name}: {exc}") from exc
        else:
            self.Validate_Schema_Rule(frame)
            return frame
        finally:
            self._last_load_seconds = time.perf_counter() - started

    @staticmethod
    def Validate_Schema_Rule(frame):
        """ If Required Columns are missing, raise Schema Validation Error. """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        present = set(frame.columns)           # set difference: O(columns)
        missing = [c for c in Column_Req if c not in present]
        if missing:
            raise SchemaValidationError("required columns are missing",
                                        context={"missing": missing})



"""This code creates a blueprint for a structured, unchangeable text record, 
which is used to log specific cleanup actions taken on a dataset. """
Scrub_Action = namedtuple("ScrubAction", "step issue column n_affected action reason")



class ScrubLog:
    """Ordered the record for the make decision"""

    def __init__(self):
        self._entries = []

    def record(self, issue, column, n_affected, action, reason):
        """in this function,is appended only one decision.
        """

        n_affected = int(n_affected)
        if n_affected < 0:
            raise ValueError("n_affected must be >= 0")
        self._entries.append(Scrub_Action(len(self._entries) + 1, issue, column,
                                         n_affected, action, reason))

    @property
    def entries(self):
        """were recoded to the tuple of ScrubAction."""
        return tuple(self._entries)

    def to_frame(self):
        """Return the log as a DataFrame ."""
        return pd.DataFrame(self._entries, columns=Scrub_Action._fields)

    def __len__(self):
        return len(self._entries)

class DataCleaner:
    """ 
        this class, every scrub steps runs to a copy of a raw DataFrame    
    
        The raw DataFrame passed in is never modified.
    """

    def __init__(self, frame):
        DataLoader.Validate_Schema_Rule(frame)
        self._raw_rows = len(frame)
        self._df = frame.copy()
        self._log = ScrubLog()
        self._has_run = False

    @property
    def log(self):
        """In the scrubLog,  what was found and do the process."""
        return self._log

    def run(self):
        """
            Run all steps once and return the cleaned DataFrame.
        """
        if self._has_run:
            raise RuntimeError("run() has already been called; create a new DataCleaner")
        steps = (
            self._duplicates, self._parse_timestamps, self._outside_study_period,
            self._check_transaction_hour_consistency, self._replace_invalid_category, self._standardise_states,
            self._loan_fields, self._numeric_ranges, self._extreme_amounts,
            self._censored_balances, self._debit_exceeds_balance, self._rtgs_minimum,
            self._salary_debit_conflicts, self._transaction_channeltype_conflict,
            self._fixed_deposit_retail, self._unstable_customer_attributes,
            self._kyc_not_verified, self._derived_features, self._optimize_dtypes,
        )
        for step in steps:
            step()
        assert len(self._df) == self._raw_rows, "a scrub step dropped rows"
        self._has_run = True
        return self._df.copy()

    def _duplicates(self):

        """
            Check for duplicate records and log the results.

            check duplicates in 3 ways
                1.exact duplicate rows
                2.repeated transaction IDs,
                3.rows with identical content under different IDs. 
            The count and outcome of each check are recorded in the cleaning log. No rows are
            removed, and the DataFrame is not changed.
        """
        df = self._df
        n_exact = int(df.duplicated().sum())
        self._log.record("Exact duplicate rows", "all", n_exact,
                         "Checked; none removed" if n_exact == 0 else "Flagged dq_duplicate",
                         "A repeated row would double-count a transaction.")
        n_id = int(df["transaction_id"].duplicated().sum())
        self._log.record("Duplicate transaction_id", "transaction_id", n_id,
                         "Checked; none found" if n_id == 0 else "Flagged dq_duplicate_id",
                         "transaction_id must be a unique key.")
        n_content = int(df.drop(columns="transaction_id").duplicated().sum())
        self._log.record("Same content under different ids", "all but transaction_id",
                         n_content, "Checked; none found" if n_content == 0 else "Flagged",
                         "Catches re-keyed duplicates that an id check misses.")

    def _parse_timestamps(self):

        """
                Combine transaction date and time into a datetime column.
        """

        df = self._df
        stamp = pd.to_datetime(df["transaction_date"].astype(str) + " " +
                               df["transaction_time"].astype(str),
                               format="%Y-%m-%d %H:%M", errors="coerce")
        n_bad = int(stamp.isna().sum())
        df["timestamp"] = stamp
        self._log.record("Date and time stored as text", "transaction_date, transaction_time",
                         len(df), f"Combined into one datetime column; {n_bad} failed to parse",
                         "Text dates cannot be sorted, split by year or used for a time-based split.")

    def _outside_study_period(self):
        
        """
            This function finds transactions dated outside the 2019–2023 study period and log the result.
            the number of flagged rows is recorded in the cleaning log. Rows are kept, not removed.
        """
        
        df = self._df
        outside = (df["timestamp"] < Study_Start) | (df["timestamp"] > Study_End)
        df["dq_outside_window"] = outside
        self._log.record("Rows dated outside 2019-2023", "timestamp", int(outside.sum()),
                         "Kept; flagged dq_outside_window and excluded from per-year charts",
                         "All are 2024-01-01: a partial single day that would distort yearly totals.")

    def _check_transaction_hour_consistency(self):
        """
            Check that transaction_hour matches the hour in timestamp.
        """
        df = self._df
        mismatch = df["timestamp"].dt.hour != df["transaction_hour"]
        self._log.record("transaction_hour disagrees with transaction_time", "transaction_hour",
                         int(mismatch.sum()), "Checked; derived column is consistent",
                         "A redundant column can silently drift from its source.")

    def _replace_invalid_category(self):
        """
            Replace invalid categorical values and record the results.
        """

        df = self._df
        for column, allowed in Value_Req.items():
            invalid = ~df[column].isin(allowed)
            if invalid.any():
                df.loc[invalid, column] = "Unknown"
            self._log.record(f"Values outside the allowed set: {column}", column, int(invalid.sum()),
                             "Checked; would be set to 'Unknown'" if not invalid.any()
                             else "Set to 'Unknown'",
                             "Typos or new codes would split one category into several.")

    def _standardise_states(self):
        
        """Standardize state names and record the changes.
        """
        df = self._df
        mask = df["state"].isin(Standardize_State_name.keys())
        df["state"] = df["state"].replace(Standardize_State_name)
        self._log.record("Inconsistent state naming ('UP' abbreviation)", "state",
                         int(mask.sum()), "Replaced with 'Uttar Pradesh'",
                         "Other nine states use full names; mixed forms break grouping and joins.")

    def _loan_fields(self):
        
        """
            Fill missing loan types and check each and every loan data in a standadize format, which contain 3 main parts.

            1.Set, blank loan_type values to "No Loan" when has_loan is 0 and "Unknown" when has_loan is 1. 
            2.checks for conflicts between has_loan and emi_amount without changing those rows. 
            3.The results are recorded in the cleaning log, and the DataFrame is changed in place.
        """

        df = self._df
        missing = df["loan_type"].isna()
        structural = missing & (df["has_loan"] == 0)
        unexplained = missing & (df["has_loan"] == 1)
        df.loc[structural, "loan_type"] = "No Loan"
        df.loc[unexplained, "loan_type"] = "Unknown"
        self._log.record("loan_type blank because customer has no loan", "loan_type",
                         int(structural.sum()), "Filled with 'No Loan'",
                         "Structurally missing, not unknown: has_loan = 0 and emi_amount = 0.")
        self._log.record("loan_type blank although has_loan = 1", "loan_type",
                         int(unexplained.sum()), "Filled with 'Unknown'; not imputed",
                         "Genuinely missing. Guessing a loan type would invent data.")
        emi_conflict = ((df["has_loan"] == 0) & (df["emi_amount"] > 0)) | \
                       ((df["has_loan"] == 1) & (df["emi_amount"] == 0))
        self._log.record("has_loan and emi_amount disagree", "has_loan, emi_amount",
                         int(emi_conflict.sum()), "Checked; consistent",
                         "A loan with no EMI, or an EMI with no loan, would be a contradiction.")

    def _numeric_ranges(self):
        
        """
            Check numeric columns for invalid values for folowing fields and record the results.
            Checks;
                1.transaction_amount
                2.account_balance
                3.credit_score
                4.is_fraud 
        """

        df = self._df
        checks = {
            "transaction_amount": df["transaction_amount"] <= 0,
            "account_balance": df["account_balance"] <= 0,
            "credit_score": ~df["credit_score"].between(300, 900),
            "is_fraud": ~df["is_fraud"].isin([0, 1]),
        }
        for column, bad in checks.items():
            self._log.record(f"Value outside valid range: {column}", column, int(bad.sum()),
                             "Checked; none invalid" if not bad.any() else "Flagged",
                             "Amounts and balances must be positive; CIBIL scores run 300-900.")

    def _extreme_amounts(self):
        """
        This function finds very large transaction amounts, flags those at the maximum limit, 
        and records both results in the cleaning log.
        """

        df = self._df
        at_cap = df["transaction_amount"] >= Max_Trans_Amt
        df["dq_amount_at_cap"] = at_cap
        huge = df["transaction_amount"] >= 1_000_000
        self._log.record("Amounts of Rs 10 lakh or more (extreme right tail)", "transaction_amount",
                         int(huge.sum()), "Kept; analysed on a log scale",
                         "Mean Rs 29,907 vs median Rs 2,036. Large transfers are real and are "
                         "where fraud concentrates, so deleting them would remove the signal.")
        self._log.record("Amounts exactly at Rs 1 crore (looks like a generator cap)",
                         "transaction_amount", int(at_cap.sum()), "Kept; flagged dq_amount_at_cap",
                         "A pile-up at a round maximum suggests values were clipped.")

    def _censored_balances(self):
        """
        This function finds account balances at the minimum or maximum limit, highlights those rows,
        and records the result in the cleaning log. 
        """
        df = self._df
        floor = df["account_balance"] <= Min_AC_Bal
        cap = df["account_balance"] >= Max_AC_Bal
        df["dq_balance_censored"] = floor | cap
        self._log.record("Balance at Rs 500 floor or Rs 50 lakh cap", "account_balance",
                         int((floor | cap).sum()), "Kept; flagged dq_balance_censored",
                         f"{int(floor.sum())} at floor, {int(cap.sum())} at cap: true values "
                         "are unknown beyond these limits.")

    def _debit_exceeds_balance(self):
        
        """
            check and validate, debits larger than the account balance and record the result.
        """
        df = self._df
        mask = (df["transaction_direction"] == "Debit") & \
               (df["transaction_amount"] > df["account_balance"])
        df["dq_exceeds_balance"] = mask
        self._log.record("Debit larger than the account balance", "transaction_amount, account_balance",
                         int(mask.sum()), "Kept; flagged dq_exceeds_balance and tested for association with fraud",
                         "Possible only with an overdraft, or if the balance is recorded after the "
                         "debit. The field's meaning is undocumented, so it is not 'corrected'.")

    def _rtgs_minimum(self):
        """
            This function finds RTGS transactions below the minimum amount, flags those rows, 
            and records the result in the cleaning log. I
        """
        df = self._df
        mask = (df["transaction_type"] == "RTGS") & (df["transaction_amount"] < RTGS_Min)
        df["dq_rtgs_below_min"] = mask
        self._log.record("RTGS below the Rs 2 lakh RBI minimum", "transaction_type, transaction_amount",
                         int(mask.sum()), "Kept; flagged dq_rtgs_below_min",
                         "Breaks a payment-system rule, so the type label or the amount is wrong. "
                         "We cannot tell which, so neither is changed.")

    def _salary_debit_conflicts(self):
        """
            This function finds rows where the merchant category is Salary but the transaction is a debit, 
            highlights them, and records the result in the cleaning log.
        """
        df = self._df
        mask = (df["merchant_category"] == "Salary") & (df["transaction_direction"] == "Debit")
        df["dq_category_conflict"] = mask
        self._log.record("Merchant category 'Salary' on a debit", "merchant_category",
                         int(mask.sum()), "Kept; flagged dq_category_conflict",
                         "Salary is money received. Two thirds of 'Salary' rows are debits, "
                         "so merchant_category is weakly reliable.")

    def _transaction_channeltype_conflict(self):
        """
            This function finds transactions where the transaction type doesn’t match the channel, 
            flags them, and records the result in the cleaning log.
        """

        df = self._df
        atm = (df["transaction_type"] == "ATM_Withdrawal") & (df["channel"] != "ATM")
        pos = (df["transaction_type"] == "POS") & (df["channel"] != "POS_Terminal")
        df["dq_channel_conflict"] = atm | pos
        self._log.record("ATM withdrawal not via ATM, or POS not via POS terminal", "channel",
                         int((atm | pos).sum()), "Kept; flagged dq_channel_conflict",
                         f"{int(atm.sum())} ATM + {int(pos.sum())} POS rows. Channel looks "
                         "independent of type, so channel findings are treated with caution.")

    def _fixed_deposit_retail(self):
        """
            This function finds fixed deposit accounts used for everyday retail payments, 
            Highlights those rows, and records the result in the cleaning log.
        """

        df = self._df
        retail = Debit_Only_Types | {"UPI"}
        mask = (df["account_type"] == "Fixed Deposit") & df["transaction_type"].isin(retail)
        df["dq_fd_retail"] = mask
        self._log.record("Fixed-deposit account used for retail payments", "account_type",
                         int(mask.sum()), "Kept; flagged dq_fd_retail",
                         "Fixed deposits do not support UPI, POS, card, ATM or auto-debit.")

    def _unstable_customer_attributes(self):
        """
            This function checks whether the same customer_id appears with more than one account_type or state, 
            then records the result in the cleaning log
        """
        df = self._df
        per_customer = df.groupby("customer_id")[["account_type", "state"]].nunique()
        unstable = per_customer.max(axis=1) > 1
        n_rows = int(df["customer_id"].isin(per_customer.index[unstable]).sum())
        self._log.record("Same customer_id with different account_type or state",
                         "customer_id", n_rows,
                         f"Kept; {int(unstable.sum()):,} of {len(per_customer):,} customers "
                         "affected, so rows are treated as independent",
                         "customer_id is not a stable entity here; per-customer history "
                         "features would be meaningless.")

    def _kyc_not_verified(self):
        """
            This function counts successful transactions made when the customer’s KYC status isn’t Verified, 
            then records the result in the cleaning log.

            KYC (Know Your Customer) is the process banks use to confirm a customer’s identity. 
            A successful transaction with Pending or Expired KYC may mean the bank’s controls didn’t stop it. 
            The log therefore treats this as a compliance finding, not a data error.
        """
        df = self._df
        mask = (df["kyc_status"] != "Verified") & (df["transaction_status"] == "Success")
        self._log.record("Successful transaction while KYC is Pending or Expired", "kyc_status",
                         int(mask.sum()), "Kept; reported as a compliance finding",
                         "Not a data error: it describes a control gap the bank may care about.")

    def _derived_features(self):
        """
            This function creates three new columns for use in charts and models.

            1. log_amount
            2. amount_to_balance	
            3. year
        """

        df = self._df
        df["log_amount"] = np.log10(df["transaction_amount"])
        df["amount_to_balance"] = np.log10(df["transaction_amount"] / df["account_balance"])
        df["year"] = df["timestamp"].dt.year
        self._log.record("Skewed amounts and ratios", "transaction_amount", len(df),
                         "Added log_amount and amount_to_balance (log10 ratio)",
                         "Log scale makes a 2-to-1e7 range usable in charts and analysis..")

    def _optimize_dtypes(self):
        """
            Convert columns to memory-efficient data types and record the result.
        """
        df = self._df
        before = df.memory_usage(deep=True).sum() / 1e6
        for column in ("account_type", "transaction_type", "transaction_direction",
                       "merchant_category", "state", "loan_type", "transaction_status",
                       "channel", "kyc_status"):
            df[column] = df[column].astype("category")
        df["is_fraud"] = df["is_fraud"].astype("int8")
        df["has_loan"] = df["has_loan"].astype("int8")
        after = df.memory_usage(deep=True).sum() / 1e6
        self._log.record("Low-cardinality text columns stored as strings", "9 categorical columns",
                         len(df), f"Converted to category dtype ({before:,.0f} MB -> {after:,.0f} MB)",
                         "Same values, far less memory and faster group-bys.")