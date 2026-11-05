

"""Main Project File"""

import pandas as pd
from DataLoad import DataLoader
from exceptions import DataFileError


        
# read dataset

file_path="F:/Nuwan/MSc/Programming For Data Science/Coursework/Project/Coursework - programming for DS/IndianBankingTransactions.csv"

try:
    loader = DataLoader(file_path)
    rows = loader.load()
except ValueError as error:
    print("Invalid input:", error)
except DataFileError as error:
    print("Data problem:", error)
else:
    print("Rows loaded:", rows)
    print(loader.get_data().head(100)) # give 1st 100 rows
finally:
    print("Program finished")
    


    want to change as new one that means without machine learning
