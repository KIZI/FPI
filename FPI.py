import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import fim

class FPI():
    """
    Algorithm proposed by:
    J. Kuchar, V. Svatek: Spotlighting Anomalies using Frequent Patterns, Proceedings of the KDD 2017 Workshop on Anomaly Detection in Finance, Halifax, Nova Scotia, Canada, PMLR, 2017.
    FPI stands for Frequent Pattern Isolation

     Parameters:

     -----------
     support:float
     data: pandas dataFrame
     mlen: integer

    """

    def __init__(self, data, support=0.1, mlen=0.5):
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")
        if mlen < 0:
            raise Exception("maximum length must be greater than 0")

        if not isinstance(data, pd.DataFrame):
            raise Exception("Data must be Pandas DataFrame")

        self.support = support * 100
        self.data = data
        self.mlen = mlen

    def build(self):

        """
        Takes variables from constructor and outputs
        anomaly scores for each row/observation as a pandas data frame
        """

        # create variables which hold number of rows and columns
        rows = len(self.data.index)
        cols = len(self.data.columns)

        # default value of mlen parameter is equal to number of columns
        if self.mlen == 0.5:
            self.mlen = cols

        # adding column name to each row
        data2 = pd.DataFrame({col:str(col)+'=' for col in self.data}, index=self.data.index) + self.data.astype(str)

        # transforming dataset to list of lists
        records = []
        for i in range(0, rows):
            records.append([str(data2.values[i, j]) for j in range(0, cols)])

        # creating transaction dataset
        print("creating transactions")
        te = TransactionEncoder()
        oht_ary = te.fit(records).transform(records, sparse=True)

        # creating sparse data frame from transaction encoder
        sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)

        # using apriori to find frequent itemsets
        print("using apriori")
        apr = fim.apriori(records, target="s", supp=self.support, zmax=self.mlen, report="s")

        # adding new column length of the rule

        frequent_itemsets = pd.DataFrame(apr)
        frequent_itemsets['length'] = frequent_itemsets[0].apply(lambda x: len(x))

        # creating a numpy array of lengths and qualities so operation such as multiplication can be done
        fiLenghts = np.array([frequent_itemsets['length']])
        fiQualities = np.array([frequent_itemsets[1]])

        # converting itemsets to frozensets so subsetting can be done
        items_list = []
        fi = frequent_itemsets[0]
        for i in fi:
            items_frozen = frozenset(i)
            items_list.append(items_frozen)

        # converting transactions to frozensets
        transactions = []
        for i in records:
            i = frozenset(i)
            transactions.append(i)

        # list that will temporarily store coverages
        tmp = []

        print("computing coverages")
        # comparing each transaction with itemsets
        for i in items_list:
            for i2 in transactions:
                if i.issubset(i2):
                    tmp.append(1)
                else:
                    tmp.append(0)

        # converting coverages to numpy array
        coverages = np.array([tmp])

        # converting coverages to valid shape and creating transpose matrix
        fiCoverages = coverages.reshape(len(frequent_itemsets), rows)
        fiCoveragesT = np.transpose(fiCoverages)
        fiQualitiesT = np.transpose(fiQualities)

        # compute basic score for each coverage
        result = 1 / (fiQualitiesT * fiLenghts)

        # create matrix with results on diagonal
        result2 = np.diagonal(result)
        shape = (len(frequent_itemsets), len(frequent_itemsets))

        # it was necessary to create matrix with zeros to have matrix with particular shape with values only on the diagonal
        diagonalHelper = np.zeros(shape)
        np.fill_diagonal(diagonalHelper, result2)

        # matrix multiplication
        scores = np.matmul(fiCoveragesT, diagonalHelper)

        # prepare  items for subsetting
        data_items = sparse_df.columns.values.tolist()

        dataItems = pd.DataFrame(data_items)

        # coverage of each data item
        dataItemsList = []

        # converting to frozenset so subsetting can be done
        for i in range(0, len(dataItems.values)):
            dataItemsList.append(frozenset([str(dataItems.values[i, j]) for j in range(0, 1)]))

        dataItemsCoverage = []

        # subsetting columns with items
        for i in dataItemsList:
            for i2 in items_list:
                if i2.issubset(i):
                    dataItemsCoverage.append(1)
                else:
                    dataItemsCoverage.append(0)

        # converting coverages to numpy array
        dataItemsCoverageArr = np.array([dataItemsCoverage])

        tmp4 = dataItemsCoverageArr.reshape(len(dataItems.values), len(frequent_itemsets))

        # variable that stores sum of columns
        colSums = np.array(self.data.count(axis=1))

        # variable that stores sum of rows
        rowSums = np.array([fiCoveragesT.sum(axis=1)])

        # preparing parts of the equation
        part1 = np.matmul(fiCoveragesT, np.transpose(tmp4))

        part2 = part1.sum(axis=1)

        # compute how many items of each transaction is not covered by appropriate frequent itemsets
        fiC = colSums - part2


        # compute final score as a mean value of scores and penalizations: (sum of scores + penalization*number of transactions)/(number of scores + penalization)
        scorings = (scores.sum(axis=1) + fiC * rows) / (rowSums + fiC)

        # creating pandas data frame with Scores column
        columnOutput = ["Scores"]
        output = pd.DataFrame(data=np.transpose(scorings), index=data2.values, columns=columnOutput, dtype=object)

        # print anomaly scores for each row/observation
        print(output)

        # returns maximum value of anomaly scores
        print(output[output['Scores'] == output['Scores'].max()])

        return output










