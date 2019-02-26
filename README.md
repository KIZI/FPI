# FPI
Frequent Pattern Isolation algorithm 

Python implementation of this algorithm based on R implementation which you can find here:
https://github.com/jaroslav-kuchar/fpmoutliers

## Usage

Basic example:

Note that you have to use pandas DataFrame as a first parametr and float as a support. Support can't be greater than 1 and less than 
0 or you'll get an error.

```python
from FPI import FPI
import pandas as pd

data = pd.read_csv("data/customerData.csv")
abc = FPI(data, 0.3)

FPI.build(abc)
```

## Output
You will get an output with anomaly scores for each row/observation and minimum and maximum anomaly score.
