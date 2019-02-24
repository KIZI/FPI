from FPI import FPI
import pandas as pd

data = pd.read_csv("data/customerData.csv")

abc = FPI(data, support=0.3)

FPI.build(abc)