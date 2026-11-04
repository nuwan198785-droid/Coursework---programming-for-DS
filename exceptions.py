"""This File gives the Exception of the banking Tranaction system analyse project"""

""" 
Built-in exceptions keep their usual meaning:
    * ValueError        - malformed input (bad amount, unknown category, bad id)
    * TypeError         - input of the wrong type
    * FileNotFoundError - a path that does not exist
"""

class BankingDataError(Exception):
    """
    The error rises when the redable description is wrong.
    eg: ids, amounts, column names , if those are failure to get informative, gives error message
    """

    def __init__(self, message, *, context=None):
        super().__init__(message)
        self.context = dict(context or {})

    def __str__(self):
        base = super().__str__()
        if not self.context:
            return base
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{base} ({details})"


class BankError(Exception):
    """This exception occurs when any operations would break the bank rules"""

class DataFileError(Exception):
    """This exception occurs when the CSV file is missing or does not have column data."""

class InsufficientBalanceError(BankError):
    """This exception occurs when a withdrawal would break the account's balance rule."""

class SchemaValidationError(BankingDataError):
    """ This exception occurs when data file's columns do nor match the expected schema."""

class DataIntegrityError(BankingDataError):
    """this exception rises when the raw extract's does not match the recorded value"""

class TransactionRuleError(BankingDataError):
    """This exception occurs when a transaction breaks a banking business rule"""

class InsufficientFundsError(TransactionRuleError):
    """A debit would take an account below its permitted overdraft limit."""



















