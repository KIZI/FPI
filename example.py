from FPI import FPI
import pandas as pd


data = pd.read_csv("data/customerData.csv")

fpi = FPI(data, 0.3)

FPI.build(fpi)

