"""
    Unit tests for the Fraud Detection Project.

    Checks that the project s core functionality behaves as expected, including
    handling of valid inputs, edge cases, and invalid data.

    To run:
        Run from the project root: python -m pytest -v
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataLoad import Column_Req,DataCleaner,DataLoader
from BankTransaction import CreditTransaction,DebitTransaction,Transaction
from BankAccount import SavingsAccount,CurrentAccount,FixedDepositAccount,open_account
from Analysis import FraudAnalyzer,wilson_interval
from exceptions import DataIntegrityError,InsufficientFundsError,SchemaValidationError,TransactionRuleError


WHEN = datetime(2023, 5, 1, 10, 30)


def debit(n=1, amount=1_000.0, txn_type="UPI"):
    """Create a sample Debit Transaction for testing purpose.
    """
    return DebitTransaction(f"TXN{n:09d}", "CUST000001", WHEN, txn_type, amount, "Mobile_App")


def credit(n=1, amount=1_000.0, txn_type="NEFT"):
    """Create a sample Credit Transaction for testing purpose.
    """
    return CreditTransaction(f"TXN{n:09d}", "CUST000001", WHEN, txn_type, amount, "Web")


def sample_row(**overrides):
    """Create a sample transaction row for use in tests
    """
    row = {
        "transaction_id": "TXN000000001", "customer_id": "CUST000001",
        "transaction_date": "2023-05-01", "transaction_time": "10:30",
        "account_type": "Savings", "transaction_type": "UPI", "transaction_amount": 500.0,
        "transaction_direction": "Debit", "account_balance": 10_000.0,
        "merchant_category": "Retail", "state": "UP", "credit_score": 700, "has_loan": 0,
        "loan_type": None, "emi_amount": 0.0, "transaction_status": "Success",
        "channel": "Mobile_App", "kyc_status": "Verified", "is_fraud": 0, "transaction_hour": 10,
    }
    row.update(overrides)
    return row


# Testing transactions
def test_debit_and_credit_are_polymorphic():
    """Check that credit and debit transactions apply their own signed amounts.
    """
    account = SavingsAccount("ACC1", opening_balance=5_000)
    account.Post_Transaction(credit(1, 2_000))
    account.Post_Transaction(debit(2, 1_500))
    assert account.balance == pytest.approx(5_500)
    assert [t.signed_amount() for t in account.transactions] == [2_000, -1_500]    # Assert: Credits are positive and debits are negative


@pytest.mark.parametrize("bad_amount", [0, -10, float("nan"), float("inf")])
def test_malformed_amount_raises_value_error(bad_amount):
    """
    This is a parametrized unit test. It runs the same test with different invalid amounts 
    and checks that each one raises a ValueError.
    """
    with pytest.raises(ValueError):
        debit(amount=bad_amount)


def test_non_numeric_amount_raises_type_error():
    """this function check that creating a debit transaction with a non-numeric amount fails or not."""
    with pytest.raises(TypeError):
        debit(amount="1000")


def test_malformed_transaction_id_raises_value_error():
    """this function check that a badly formatted transaction ID raises a error when clearing."""
    with pytest.raises(ValueError, match="malformed"):
        DebitTransaction("TX1", "CUST000001", WHEN, "UPI", 10.0)


def test_credit_cannot_be_a_debit_only_type():
    """thic function check that a credit transaction can t use a debit type."""
    with pytest.raises(ValueError, match="not allowed"):
        credit(txn_type="ATM_Withdrawal")


def test_transaction_is_abstract():
    """this function check that the abstract Transaction class can t be used directly."""
    with pytest.raises(TypeError):
        Transaction("TXN000000001", "CUST000001", WHEN, "UPI", 10.0)

"""
def test_from_record_builds_the_right_subclass():
"""

"""This menthod check that from_record uses the direction to choose the transaction class.
"""

"""
    assert isinstance(Transaction.from_record(sample_row()), DebitTransaction)
    assert isinstance(Transaction.from_record(sample_row(transaction_direction="Credit")),
                      CreditTransaction)
    with pytest.raises(ValueError):
        Transaction.from_record(sample_row(transaction_direction="Sideways"))
"""

def test_risk_flags():
    """This method check that the expected risk flags are returned for each transaction."""
    rtgs = debit(amount=150_000, txn_type="RTGS")
    assert rtgs.get_risk_flags() == ("HIGH_RISK_TYPE", "RTGS_BELOW_MINIMUM")
    assert debit(amount=250_000, txn_type="NEFT").get_risk_flags() == ("HIGH_VALUE",)
    assert debit(amount=100).get_risk_flags() == ()


# Testing accounts
def test_insufficient_funds_is_custom_and_leaves_balance_unchanged():
    """this function check that a debit exceeding available funds is rejected or not."""
    account = SavingsAccount("ACC1", opening_balance=1_000)
    with pytest.raises(InsufficientFundsError) as info:
        account.Post_Transaction(debit(amount=5_000))
    assert "available funds" in str(info.value)
    assert account.balance == 1_000 and len(account) == 0


def test_current_account_overrides_overdraft():
    """this method check that a current account can go below a zero balance."""
    account = CurrentAccount("ACC2", opening_balance=1_000)
    account.Post_Transaction(debit(amount=50_000))
    assert account.balance == pytest.approx(-49_000)


def test_duplicate_id_is_rejected():
    """
    this method checks whether an account rejects the same transaction ID twice. 
    It posts one debit successfully, then tries to post another with the same ID.
    """
    account = SavingsAccount("ACC1", opening_balance=10_000)
    account.Post_Transaction(debit(1))
    with pytest.raises(TransactionRuleError, match="duplicate"):
        account.Post_Transaction(debit(1))


def test_fixed_deposit_refuses_retail_payments():
    """this fuction check that a fixed deposit blocks retail payments but allows NEFT."""
    account = FixedDepositAccount("FD1", opening_balance=100_000)
    with pytest.raises(TransactionRuleError, match="fixed deposits"):
        account.Post_Transaction(debit(txn_type="POS"))
    account.Post_Transaction(debit(2, txn_type="NEFT"))
    assert len(account) == 1


def test_post_many_rolls_back_the_whole_batch():
    """
    This unit test is for an all-or-nothing batch. post_many() tries to post three debits, but the third is too large. 
    The test checks that the whole batch is cancelled, including the first two debits, which would have succeeded on their own.
    """
    account = SavingsAccount("ACC1", opening_balance=2_000)
    batch = [debit(1, 500), debit(2, 500), debit(3, 50_000)]      
    with pytest.raises(InsufficientFundsError):
        account.post_many(batch)
    assert account.balance == 2_000
    assert len(account) == 0 and "TXN000000001" not in account
    assert account.batches_attempted == 1                          


def test_open_account_unknown_type():
    """
    This is a unit test for open_account()
    it takes an account type as text and creates the matching account object. 
    The test checks that an unknown type is rejected and a known type creates the correct class.
    """
    with pytest.raises(ValueError):
        open_account("Crypto", "X1")
    assert isinstance(open_account("Fixed Deposit", "X2"), FixedDepositAccount)


#  loading/cleaning
def test_loader_rejects_wrong_extension(tmp_path):
    """
    This is a unit test that checks whether DataLoader accepts only CSV files.
    """
    bad = tmp_path / "data.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match=".csv"):
        DataLoader(bad)


def test_loader_missing_file():
    """
    This is a unit test that checks what happens when DataLoader is given a file that doesnt exist.
    """
    with pytest.raises(FileNotFoundError):
        DataLoader("does_not_exist.csv")


def test_loader_detects_changed_file(tmp_path):
    """
    This is a unit test for the DataLoader integrity check, which detects whether a file’s contents differ from what was expected.
    """
    path = tmp_path / "raw.csv"
    pd.DataFrame([sample_row()]).to_csv(path, index=False)
    with pytest.raises(DataIntegrityError):
        DataLoader(path, Expected_sha256_Val="0" * 64).load()


def test_schema_error_lists_missing_columns(tmp_path):
    """
    This is a unit test that checks whether DataLoader detects a missing required column.
    """
    path = tmp_path / "raw.csv"
    pd.DataFrame([sample_row()]).drop(columns=["is_fraud"]).to_csv(path, index=False)
    with pytest.raises(SchemaValidationError, match="is_fraud"):
        DataLoader(path).load()


def test_cleaner_keeps_rows_and_does_not_touch_input():
    """
    This is a unit test for the DataCleaner class. It runs the cleaner on a small two-row DataFrame 
    and checks that the output is cleaned correctly, 
    the original data stays unchanged, the steps are logged, and the cleaner can’t be run twice.
    """
    raw = pd.DataFrame([sample_row(), sample_row(transaction_id="TXN000000002",
                                                 has_loan=1, emi_amount=900.0)])
    raw_copy = raw.copy()
    cleaner = DataCleaner(raw)
    clean = cleaner.run()
    pd.testing.assert_frame_equal(raw, raw_copy)                   
    assert len(clean) == len(raw)
    assert list(clean["loan_type"]) == ["No Loan", "Unknown"]
    assert set(clean["state"]) == {"Uttar Pradesh"}
    assert len(cleaner.log) > 10
    with pytest.raises(RuntimeError):
        cleaner.run()


def test_required_columns_is_a_tuple():
    """
    This is a unit test for the REQUIRED_COLUMNS constant. It checks that the constant is a tuple and contains exactly 20 items,
    """
    assert isinstance(Column_Req, tuple) and len(Column_Req) == 20


#  analysis
def test_wilson_interval_contains_rate():
    """
    This is a unit test for wilson_interval(), which calculates a Wilson score confidence interval
    Check the interval s bounds and that a total of zero is rejected.
    """
    low, high = wilson_interval(9, 1_000)
    assert low < 0.009 < high
    with pytest.raises(ValueError):
        wilson_interval(5, 0)


def test_analyzer_rejects_bad_column():
    """
    This is a unit test for the FraudAnalyzer class. It creates a small DataFrame with one normal and one fraudulent transaction, 
    checks that the overall fraud rate is 0.5, and checks that asking for a column that doesn’t exist raises a ValueError.
    """
    frame = pd.DataFrame({"is_fraud": [0, 1], "transaction_amount": [1.0, 2.0],
                          "timestamp": pd.to_datetime(["2020-01-01", "2023-01-01"]),
                          "transaction_type": ["UPI", "RTGS"]})
    analyzer = FraudAnalyzer(frame)
    assert analyzer.base_rate == 0.5
    with pytest.raises(ValueError):
        analyzer.fraud_rate_by("nope")
