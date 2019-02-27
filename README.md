# FPI
Frequent Pattern Isolation algorithm 

Python implementation of this algorithm based on R implementation which you can find here:
https://github.com/jaroslav-kuchar/fpmoutliers

## Usage

Basic example:

Note that you have to use pandas DataFrame as a first parameter and float as second. Second parameter (support) can't be greater than 1 and less than 0 or you'll get an error.

```python
from FPI import FPI
import pandas as pd

data = pd.read_csv("data/customerData.csv")
fpi = FPI(data, support=0.3)

FPI.build(fpi)
```

## Output
You will get an output with anomaly scores for each row/observation and minimum and maximum anomaly score.
