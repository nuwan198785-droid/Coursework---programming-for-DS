"""This File contains BankAccount and its sub catogories (types of accounts)"""


from BankTransaction import BankTransaction
from exceptions import InsufficientBalanceError


class BankAccount:
    """
    balance >=0 (banalce is private)
    can change only deposit, withdraw
    """

    def __init__(self, balance):
        """initiate an account"""

        if isinstance(balance, bool) or not isinstance(balance, (int, float)) or balance < 0:
            raise ValueError("balance must be a number >= 0")
        self.__balance = balance

    def get_balance(self):
        return self.__balance # return the available balance


    def minimum_balance(self):
        return 0


    def deposit(self, amount):
        self.__check_amount(amount) # amount must be in valid format and > 0
        self.__balance += amount


    def withdraw(self, amount):
        """
        consider abount the amount and the avalilable balance
        check the exception handling of ValueError  and InsufficientBalanceError
        """

        self.__check_amount(amount)
        if self.__balance - amount < self.minimum_balance():
            raise InsufficientBalanceError(
                f"Cannot withdraw {amount}. Balance is {self.__balance}.")
        self.__balance -= amount

    def apply_transaction(self, transaction):
        """Credit = deposit, Debit = withdraw.
        """

        if not isinstance(transaction, BankTransaction):
            raise TypeError("transaction must be a BankTransaction")

        if transaction.transaction_type == "Credit":
            self.deposit(transaction.amount)
        else:
            self.withdraw(transaction.amount)

    @staticmethod
    def __check_amount(amount):
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("amount must be a number greater than 0")


class SavingsAccount(BankAccount):
    """Savings account: must always keep at least 1000."""

    def minimum_balance(self):
        """Override: a savings account must keep 1000."""
        return 1000


class CurrentAccount(BankAccount):
    """Here,for the operational smoothness, we consider that Current account can be overdrawn by up to 5000."""

    def minimum_balance(self):
        """Override: an overdraft of 5000 is allowed."""
        return -5000
