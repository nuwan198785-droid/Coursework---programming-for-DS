"""This File work with a single bank transaction."""

class BankTransaction:
    """
        conditions
        1. amount must be greater than 0
        2. transaction type must be credit or debit
   
    """

    # tuple: fixed list of allowed values that must never change
    Valid_Types = ("Credit", "Debit")

    def __init__(self, amount, transaction_type, transaction_id="", customer_id=""):

        """in this function Transaction details are stored."""

        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("amount must be a number greater than 0")

        if transaction_type not in self.Valid_Types:
            raise ValueError("transaction_type must be 'Credit' or 'Debit'")

        if not isinstance(transaction_id, str) or not isinstance(customer_id, str):
            raise ValueError("transaction_id and customer_id must be text")

        self.__amount = amount                      
        self.__transaction_type = transaction_type
        self.__transaction_id = transaction_id
        self.__customer_id = customer_id

    @property
    def amount(self):
        return self.__amount # return the amount

    @property
    def transaction_type(self):
        return self.__transaction_type # return the trasaction type

    @property
    def transaction_id(self):
        return self.__transaction_id # return the transaction reference no

    @property
    def customer_id(self):
        return self.__customer_id # return the customer reference number

    def display_transaction(self):
        """Print the transaction details."""
        print("ID:", self.__transaction_id, "| Customer:", self.__customer_id,"| Amount:", self.__amount, "| Type:", self.__transaction_type)