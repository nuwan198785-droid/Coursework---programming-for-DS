"""
    here we implement the Transaction classes.

    Transaction is an abstract base class. DebitTransaction and CreditTransaction
    inherit from it and override the behaviour that genuinely differs between
    funds leaving and funds arriving

    in this python file, every subclass offers the same interface, BankAccount can post any
    transaction with ``txn.apply_to(balance)`` without asking which kind it is
    (polymorphism).

"""
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime

from exceptions import InsufficientFundsError

"""
Allowed transaction types, stored as frozensets for fast lookups and
protection against accidental changes.
"""
Transaction_Types = frozenset({
    "UPI", "IMPS", "NEFT", "POS", "ATM_Withdrawal", "Net_Banking",
    "RTGS", "Auto_Debit", "Cheque", "Credit_Card",
})
Debit_Only_Types = frozenset({"ATM_Withdrawal", "POS", "Auto_Debit", "Credit_Card"}) # Transaction types expected to be debits.
High_Risk_Types = frozenset({"RTGS", "Cheque"}) # Transaction types treated as high risk.

RTGS_Min = 200_000.0          # assume RBI minimum for an RTGS transfer (Rs 2 lakh)
High_val_Threshold = 200_000.0  # fraud rate jumps from 0.8 % to 3.2 % at Rs 2 lakh

# Expected ID and timestamp formats.
_TXN_ID_PATTERN = re.compile(r"^TXN\d{9}$")
_CUSTOMER_ID_PATTERN = re.compile(r"^CUST\d{6}$")
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


class Transaction(ABC):
    """
        only One bank transaction implement. 
            1.create a DebitTransaction or
            2.create a CreditTransaction, or 
            3.use Transaction.from_record().

        All attributes are private so a transaction cannot be altered after it has been validated.
    """

    def __init__(self, txn_id, customer_id, timestamp, txn_type, amount, channel="Unknown"):
        self._txn_id = self._validate_id(txn_id, _TXN_ID_PATTERN, "txn_id")
        self._customer_id = self._validate_id(customer_id, _CUSTOMER_ID_PATTERN, "customer_id")
        if not isinstance(timestamp, datetime):
            raise TypeError(f"timestamp must be a datetime, got {type(timestamp).__name__}")
        self._timestamp = timestamp
        if txn_type not in self.allowed_types():
            raise ValueError(
                f"txn_type {txn_type!r} is not allowed for a {self.direction} transaction; "
                f"expected one of {sorted(self.allowed_types())}"
            )
        self._txn_type = txn_type
        self._amount = self._validate_amount(amount)
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("channel must be a non-empty string")
        self._channel = channel

    @staticmethod
    def _validate_id(value, pattern, name):
        """
            This function checks that an ID is text and has the correct format. 
            If checkind fails, it raises an error. If the ID is valid, the function does nothing and the program continues.
        """
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str, got {type(value).__name__}")
        if not pattern.match(value):
            raise ValueError(f"{name} {value!r} is malformed (expected pattern {pattern.pattern})")
        return value


    @staticmethod
    def _validate_amount(amount):
        """
            This function checks that an amount is a valid positive number, 
            then returns it as a float. If the amount is invalid, it raises an error.
        """

        # bool is a subclass of int, so reject it explicitly.
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"amount must be a number, got {type(amount).__name__}")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError(f"amount must be a finite number greater than 0, got {amount!r}")
        return float(amount)


    """
        following properties that provide controlled access to the transaction's private attributes. 
        Values are returned without being changed and cannot be reassigned.
    """
    @property
    def txn_id(self):
        """str: the unique transaction id."""
        return self._txn_id

    @property
    def customer_id(self):
        """str: the customer id."""
        return self._customer_id

    @property
    def timestamp(self):
        """datetime: when the transaction happened."""
        return self._timestamp

    @property
    def txn_type(self):
        """str: payment rail, e.g. 'UPI' or 'RTGS'."""
        return self._txn_type

    @property
    def amount(self):
        """float: the unsigned amount in rupees (always > 0)."""
        return self._amount

    @property
    def channel(self):
        """str: channel the transaction came through."""
        return self._channel

    @property
    @abstractmethod
    def direction(self):
        """str: 'Debit' or 'Credit'."""

    @classmethod
    def allowed_types(cls):
        """Return the frozenset of transaction types valid for this class."""
        return Transaction_Types

    @abstractmethod
    def signed_amount(self):
        """Return the amount with the sign it has on the account balance."""

    def apply_to(self, balance, overdraft_limit=0.0):
        """
            Return the new balance after this transaction, without changing anything.
        """
        self._validate_number(balance, "balance", allow_negative=True)
        self._validate_number(overdraft_limit, "overdraft_limit", allow_negative=False)
        return float(balance) + self.signed_amount()

    def get_risk_flags(self):
        """
            This method checks a transaction against three risk rules and returns a tuple of the flags that apply.
            Checks the transaction for 
                1. a high value
                2. a high-risk type
                3. a RTGS amount below the minimum. 
            The transaction is not changed.
        """
        flags = []
        if self._amount >= High_val_Threshold:
            flags.append("HIGH_VALUE")
        if self._txn_type in High_Risk_Types:
            flags.append("HIGH_RISK_TYPE")
        if self._txn_type == "RTGS" and self._amount < RTGS_Min:
            flags.append("RTGS_BELOW_MINIMUM")
        return tuple(flags)

    @staticmethod
    def _validate_number(value, name, allow_negative):
        """
            This function checks that a value is a valid finite number and also reject negative values. 
            If the value is invalid, it raises an error.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if not allow_negative and value < 0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")

    # ----------------------------------------------------------- dunder API
    def __eq__(self, other):
        return isinstance(other, Transaction) and other._txn_id == self._txn_id

    def __hash__(self):
        return hash(self._txn_id)

    def __repr__(self):
        return (f"{type(self).__name__}(txn_id={self._txn_id!r}, type={self._txn_type!r}, "
                f"amount={self._amount:,.2f})")


class DebitTransaction(Transaction):
    """
        in this class , this implements the funds leving from accounts
    """

    @property
    def direction(self):
        return "Debit"

    def signed_amount(self):
        return -self._amount

    def apply_to(self, balance, overdraft_limit=0.0):
        """Return the new balance, or raise InsufficientFundsError"""

        new_balance = super().apply_to(balance, overdraft_limit)
        if new_balance < -overdraft_limit:
            raise InsufficientFundsError(
                "debit exceeds available funds",
                context={"txn_id": self._txn_id, "amount": self._amount,
                         "balance": balance, "overdraft_limit": overdraft_limit},
            )
        return new_balance


class CreditTransaction(Transaction):
    """In this Class implements the funds receiving process to accounts."""

    @property
    def direction(self):
        return "Credit"

    @classmethod
    def allowed_types(cls):
        return Transaction_Types - Debit_Only_Types

    def signed_amount(self):
        return self._amount


# dict dispatch: direction string -> class. Defined after the classes exist.
_DIRECTION_TO_CLASS = {"Debit": DebitTransaction, "Credit": CreditTransaction}

#print("ID:", self.__transaction_id, "| Customer:", self.__customer_id,"| Amount:", self.__amount, "| Type:", self.__transaction_type)