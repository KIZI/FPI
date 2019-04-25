import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import fim
import time
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix, coo_matrix
from sklearn.metrics import accuracy_score, confusion_matrix

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

    def __init__(self, data, support=0.1, mlen=0):
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")
        if mlen < 0:
            raise Exception("maximum length must be greater than 0")

        if not isinstance(data, pd.DataFrame):
            raise Exception("Data must be Pandas DataFrame")

        self.support = support * 100
        self.data = data
        self.mlen = mlen


    def fit(self):

        """
        Takes variables from constructor and outputs
        anomaly scores for each row/observation as a pandas data frame
        """

        # create variables which hold number of rows and columns
        rows = len(self.data.index)
        cols = len(self.data.columns)

        # default value of mlen parameter is equal to number of columns
        if self.mlen == 0:
            self.mlen = cols

        # adding column name to each row
        self.data = pd.DataFrame({col:str(col)+'=' for col in self.data}, index=self.data.index) + self.data.astype(str)

        # transforming dataset to list of lists
        records = []
        for i in range(0, rows):
            records.append([str(self.data.values[i, j]) for j in range(0, cols)])

        # creating transaction dataset
        print("Creating transactions from a dataset")
        t = time.process_time()
        te = TransactionEncoder()
        oht_ary = te.fit(records).transform(records, sparse=True)
        elapsed_time = time.process_time() - t
        print("Transactions created in: "+str(elapsed_time))
        # creating sparse data frame from transaction encoder
        sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)

        # using apriori to find frequent itemsets
        supp = self.support / 100
        print("Running apriori with settings: support={}, maxlen={}".format(supp, self.mlen))
        t = time.process_time()
        apr = fim.apriori(records, target="s", supp=self.support, zmax=self.mlen, report="s")
        elapsed_time = time.process_time() - t
        print("Apriori finished in: "+str(elapsed_time))

        # adding new column length of the rule
        frequent_itemsets = pd.DataFrame(apr)
        print("Computing fiLengths")
        t = time.process_time()
        frequent_itemsets['length'] = frequent_itemsets[0].apply(lambda x: len(x))

        # creating a matrix of lengths and qualities so operation such as multiplication can be done
        fiLenghts = coo_matrix([frequent_itemsets['length']], dtype=np.int8)
        elapsed_time = time.process_time() - t
        print("Computing fiLengths finished in: " + str(elapsed_time))
        print("Computing fiQualities")
        t = time.process_time()
        fiQualities = coo_matrix([frequent_itemsets[1]], dtype=np.float16)
        elapsed_time = time.process_time() - t
        print("Computing fiQualities finished in: " + str(elapsed_time))
        # converting itemsets to frozensets so subsetting can be done
        print("Converting to datasets frozensets and computing coverages")
        t = time.process_time()
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
        print("Computing coverages")
        # comparing each transaction with itemsets
        for i in items_list:
            for i2 in transactions:
                if i.issubset(i2):
                    tmp.append(1)
                else:
                    tmp.append(0)

        # converting coverages to numpy array
        coverages = np.array([tmp])
        elapsed_time = time.process_time() - t
        print("Computing coverages finished in: "+str(elapsed_time))
        # converting coverages to valid shape and creating transpose matrix
        fiCoveragesArr = coverages.reshape(len(frequent_itemsets), rows)
        fiCoveragesArrT = np.transpose(fiCoveragesArr)
        fiCoverages = csr_matrix(fiCoveragesArrT)

        # multiply lengths and qualities
        t = time.process_time()
        print("Multiplication of qualities and lengths")
        result1 = coo_matrix(fiQualities.tocsc().T.multiply(fiLenghts.tocsr()))
        result = 1 / result1.toarray()

        # create matrix with results on diagonal
        result2 = result.diagonal()

        # it was necessary to create matrix with zeros to have matrix with particular shape with values only on the diagonal
        diagonalHelper = np.zeros(shape=(len(frequent_itemsets), len(frequent_itemsets)))

        abcd = coo_matrix(diagonalHelper)

        abcd.setdiag(result2)
        # Compute basic scores for each coverage
        print("Computing individual scores for each coverage")
        scores = fiCoverages.dot(abcd.tocsr())
        elapsed_time = time.process_time() - t
        print("Computing results finished in: " + str(elapsed_time))
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
        dataItemsCoverageArr = np.array(dataItemsCoverage)

        tmp4 = dataItemsCoverageArr.reshape(len(dataItems.values), len(frequent_itemsets))
        tmp5 = coo_matrix(tmp4)
        # variable that stores sum of columns
        print("Computing penalizations")
        t = time.process_time()
        colSums = csr_matrix(self.data.count(axis=1))

        # variable that stores sum of rows
        rowSums = fiCoverages.sum(axis=1)

        # preparing parts of the equation
        part1 = fiCoverages.dot(tmp5.T.tocsr())

        # compute how many items of each transaction is not covered by appropriate frequent itemsets
        fiC = colSums - part1.sum(axis=1).T
        elapsed_time = time.process_time() - t
        print("Computing penalizations finished in: "+str(elapsed_time))

        # compute final score as a mean value of scores and penalizations: (sum of scores + penalization*number of transactions)/(number of scores + penalization)
        print("Computing scores for each row")
        t = time.process_time()
        scorings = (scores.sum(axis=1) + fiC * rows) / (rowSums + fiC)
        elapsed_time = time.process_time() - t
        print("Computing final scores finished in: "+str(elapsed_time))
        # creating pandas data frame with Scores column
        self.data['Scores'] = scorings.diagonal().T


        # print anomaly scores for each row/observation

        #print(self.data)
        # returns maximum value of anomaly scores
        print(self.data[self.data['Scores'] == self.data['Scores'].max()])
        return self.data

    def predict(self, testData, anomalyBase):

        """
                Takes output from fit method and assign class anomaly or normal
                """

        if not isinstance(testData, pd.DataFrame):
            raise Exception("testData must be Pandas DataFrame!")

        if anomalyBase < 0:
            raise Exception("Anomaly base can't be less than 0!")

        testData["Predicted"] = np.where(testData['Scores'] >= anomalyBase, "anomaly", "normal")
        print(testData)
        return testData


data = pd.read_csv("test/data/trainData.csv", sep=";")
labels = pd.read_csv("test/data/trainDataLabels.csv", sep=";")

fpi = FPI(data, 0.3, 5)

fpiFit = FPI.fit(fpi)




