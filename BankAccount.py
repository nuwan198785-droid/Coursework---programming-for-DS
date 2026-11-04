"""Account classes.

    Account subclasses override behaviour, that differs by account type:

        1.SavingsAccount       
        2.CurrentAccount       
        3.FixedDepositAccount  
"""

import math
from collections.abc import Iterable

from .BankTransaction import Transaction, Transaction_Types
from .exceptions import TransactionRuleError


class BankAccount:
    """
        A ledger for one account.
        ledger account is a dedicated record that tracks every single financial transaction,including all debits and credits
    """

    Acc_Type = "Generic"
    Default_OD = 0.0
    Valid_Type = Transaction_Types

    def __init__(self, account_id, opening_balance=0.0, overdraft_limit=None):
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-empty string")
        limit = self.Default_OD if overdraft_limit is None else overdraft_limit
        self._check_number(limit, "overdraft_limit")
        if limit < 0:
            raise ValueError(f"overdraft_limit must be >= 0, got {limit!r}")
        self._check_number(opening_balance, "opening_balance")
        if opening_balance < -limit:
            raise ValueError("opening_balance is below the permitted overdraft")

        self._account_id = account_id
        self._overdraft_limit = float(limit)
        self._balance = float(opening_balance)
        self._transactions = []
        self._txn_ids = set()
        self._batches_attempted = 0

    @staticmethod
    def _check_number(value, name):
        """
            This function checks that a value is a real, finite number. If the value is invalid, it raises an error.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    @property
    def account_id(self):
        """str: the account id."""
        return self._account_id

    @property
    def balance(self):
        """float: the current balance ."""
        return self._balance

    @property
    def overdraft_limit(self):
        """float: maximum overdraft balance."""
        return self._overdraft_limit

    @property
    def transactions(self):
        """tuple: posted transactions in order."""
        return tuple(self._transactions)

    @property
    def batches_attempted(self):
        """int: how many post_many() calls have been made (whether it is successful or not)."""
        return self._batches_attempted

    def available_funds(self):
        """Return balance + overdraft limit: existing funds ."""
        return self._balance + self._overdraft_limit

    def __len__(self):
        return len(self._transactions)

    def __contains__(self, txn_id):
        return txn_id in self._txn_ids      # O(1) thanks to the set

    def _validate_transaction(self, txn):

        """
            This method checks whether an account type accepts a transaction. 
            If the transaction type isn’t allowed, it raises TransactionRuleError.
            Subclasses override this to add account-specific rules.
        """
        if txn.txn_type not in self.Valid_Type:
            raise TransactionRuleError(
                f"{self.Acc_Type} account does not accept {txn.txn_type}",
                context={"account_id": self._account_id, "txn_id": txn.txn_id},
            )

    def Post_Transaction(self, txn):
        """
            This method records one transaction on the account and returns the new balance. 
            In banking, “posting” means entering a transaction into the account’s records.

            The method runs all checks first and changes the account only after they pass. 
            If a check fails, the balance and transaction history stay unchanged.
        """
        if not isinstance(txn, Transaction):
            raise TypeError(f"expected a Transaction, got {type(txn).__name__}")
        if txn.txn_id in self._txn_ids:
            raise TransactionRuleError("duplicate transaction id",
                                       context={"txn_id": txn.txn_id})
        self._validate_transaction(txn)
        new_balance = txn.apply_to(self._balance, self._overdraft_limit)  # may raise
        # ---- commit: nothing below can fail ----
        self._transactions.append(txn)
        self._txn_ids.add(txn.txn_id)
        self._balance = new_balance
        return new_balance

    def post_many(self, transactions):
        """
        This function posts a group of transactions on an all-or-nothing basis. If every transaction succeeds, it returns the number posted. 
        If one fails, it undoes the whole batch, restores the account to its starting state, and raises the error again.

        """
        if isinstance(transactions, (str, bytes)) or not isinstance(transactions, Iterable):
            raise TypeError("transactions must be an iterable of Transaction objects")

        start_balance = self._balance
        start_length = len(self._transactions)
        posted = 0
        try:
            for txn in transactions:
                self.Post_Transaction(txn)
                posted += 1
        except (TransactionRuleError, TypeError, ValueError):
            # Roll back: remove only what this batch added (cheaper than
            # copying the whole id set up front).
            for txn in self._transactions[start_length:]:
                self._txn_ids.discard(txn.txn_id)
            del self._transactions[start_length:]
            self._balance = start_balance
            raise
        else:
            return posted
        finally:
            self._batches_attempted += 1

    def summary(self):
        """
            This function gives a dictionary which contains the summarization of the account. 
            It shows the account’s ID, type, balance, and transaction counts without changing anything.
        """

        debits = sum(1 for t in self._transactions if t.direction == "Debit")
        return {
            "account_id": self._account_id,
            "account_type": self.Acc_Type,
            "balance": round(self._balance, 2),
            "n_transactions": len(self._transactions),
            "n_debits": debits,
            "n_credits": len(self._transactions) - debits,
        }

    def __repr__(self):
        return (f"{type(self).__name__}(id={self._account_id!r}, "
                f"balance={self._balance:,.2f}, n={len(self)})")


class SavingsAccount(BankAccount):
    """Savings, Salary and NRI accounts: all types allowed, no overdraft."""
    Acc_Type = "Savings"


class CurrentAccount(BankAccount):
    """Business current account. Overrides the default overdraft limit."""
    Acc_Type = "Current"
    Default_OD = 100_000.0


class FixedDepositAccount(BankAccount):
    """
        In this class (Fixed deposit). Funds move only by bank transfer or cheque ( credit or debit),
        so card,POS, UPI, ATM and auto-debit transactions are refused.
    """
    Acc_Type = "Fixed Deposit"
    Valid_Type = frozenset({"NEFT", "RTGS", "IMPS", "Net_Banking", "Cheque"})

    def _validate_transaction(self, txn):
        """Extends the base rule with a clearer, FD-specific message."""
        try:
            super()._validate_transaction(txn)
        except TransactionRuleError:
            raise TransactionRuleError(
                "fixed deposits only accept bank transfers or cheques",
                context={"account_id": self._account_id, "txn_id": txn.txn_id,
                         "txn_type": txn.txn_type},
            ) from None


# dict dispatch from the dataset's account_type value to a class.
Account_Class = {
    "Savings": SavingsAccount,
    "Salary": SavingsAccount,
    "NRI": SavingsAccount,
    "Current": CurrentAccount,
    "Fixed Deposit": FixedDepositAccount,
}


def open_account(account_type, account_id, opening_balance=0.0):
    """
        Create the correct account subclass from the values of account_type in the dataset.

    """
    try:
        cls = Account_Class[account_type]
    except KeyError:
        raise ValueError(f"unknown account_type {account_type!r}; "
                         f"expected one of {sorted(Account_Class)}") from None
    return cls(account_id, opening_balance)
