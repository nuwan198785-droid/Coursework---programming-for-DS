"""This File gives the Exception of the banking system analyse project"""


class BankError(Exception):
    """This exception occurs when a withdrawal would break the bank rules"""


class DataFileError(Exception):
    """This exception occurs when the CSV file is missing or does not have column data."""