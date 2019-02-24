import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

class FPI():
    """
    Algorithm proposed by J.Kuchar and V.Svatek

     Parameters:

     -----------
     support:float
     data: pandas dataFrame

    """

    def __init__(self, data, support):
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")

        self.support = support
        self.data = data

    def build(self):

        """
        Takes variables from constructor and outputs
        anomaly scores for each row
        """
        rows = len(self.data.index)
        cols = len(self.data.columns)
        mlen = cols

        # transforming dataset to list of lists
        records = []
        for i in range(0, rows):
            records.append([str(self.data.values[i, j]) for j in range(0, cols)])

        # creating transaction dataset
        te = TransactionEncoder()
        oht_ary = te.fit(records).transform(records, sparse=True)

        sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)

        # using apriori to find frequent itemsets

        frequent_itemsets = apriori(sparse_df, min_support=self.support, use_colnames=True, max_len=mlen, n_jobs=1)

        # adding new column lenght of the rule
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        fiLenghts = np.array([frequent_itemsets['length']])
        fiQualities = np.array([frequent_itemsets['support']])

        items = frequent_itemsets['itemsets']
        itemsets_list = list(frequent_itemsets['itemsets'])
        itemsets = np.array([frequent_itemsets['itemsets']])

        # converting itemsets to frozensets so subsetting can be done
        items_list = []
        fi = frequent_itemsets['itemsets']
        for i in fi:
            items_frozen = frozenset(i)
            items_list.append(items_frozen)

        # converting transactions to frozensets
        transactions = []
        for i in records:
            i = frozenset(i)
            transactions.append(i)


        tmp = []

        # comparing each transacation with itemsets
        for i in items_list:
            for i2 in transactions:
                if i.issubset(i2):
                    tmp.append(1)
                else:
                    tmp.append(0)

        coverages = np.array([tmp])
        fiCoverages = coverages.reshape(len(frequent_itemsets), rows)
        fiCoveragesT = np.transpose(fiCoverages)
        fiQualitiesT = np.transpose(fiQualities)

        # compute basic score for each coverage
        result = 1 / (fiQualitiesT * fiLenghts)

        result2 = np.diagonal(result)
        shape = (len(frequent_itemsets), len(frequent_itemsets))
        diagonalHelper = np.zeros(shape)
        np.fill_diagonal(diagonalHelper, result2)

        scores = np.matmul(fiCoveragesT, diagonalHelper)

        # prepare  items for subsetting
        data_items = sparse_df.columns.values.tolist()

        dataItems = pd.DataFrame(data_items)

        dataItemsList = []
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
        dataItemsCoverageArr = np.array([dataItemsCoverage])

        tmp4 = dataItemsCoverageArr.reshape(len(dataItems.values), len(frequent_itemsets))
        tmp6 = np.transpose([tmp4.sum(axis=1)])

        colSums = np.array(self.data.count(axis=1))

        rowSums = np.array([fiCoveragesT.sum(axis=1)])

        # compute how many items of each transaction is not covered by appropriate frequent itemsets
        left = np.matmul(fiCoveragesT, np.transpose(tmp4))

        right = left.sum(axis=1)

        fiC = colSums - right

        # compute final score as a mean value of scores and penalizations: (sum of scores + penalization*number of transactions)/(number of scores + penalization)
        scorings = (scores.sum(axis=1) + fiC * rows) / (rowSums + fiC)

        # print scores as pandas data frame
        columnOutput = ["Scores"]
        output = pd.DataFrame(data=np.transpose(scorings), columns=columnOutput, index=self.data.values)

        print(output)





