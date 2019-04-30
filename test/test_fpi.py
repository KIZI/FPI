import unittest
import pandas as pd
from FPI_old import FPI
import pandas.util.testing

dataset_file = pd.read_csv("data/customerData.csv")


class TestFPI(unittest.TestCase):

    def test_parameter_validation(self):
        fpi = FPI(data=dataset_file)
        listData = []
        self.assertRaises(Exception, FPI, data=listData)
        self.assertRaises(Exception, FPI, support=2)
        self.assertRaises(Exception, FPI, support=-5)
        self.assertRaises(Exception, fpi.build, None)
        self.assertRaises(Exception, FPI, mlen=-3)

    def test_building(self):
        fpi = FPI(dataset_file, 0.3)

        output = fpi.build()

        dataList = [1.66667,1.52778,1.72917,1.66667,1.9375,1.74206,1.52778,1.74206,1.52778,1.74206]
        column = ['Scores']
        indexes = [('Age-Range=Middle', 'Car=Sedan', 'Salary-Level=Low'),
        ('Age-Range=Middle', 'Car=Sedan', 'Salary-Level=High'),
        ('Age-Range=Young', 'Car=Sedan', 'Salary-Level=High'),
        ('Age-Range=Middle', 'Car=Sedan', 'Salary-Level=Low'),
       ('Age-Range=Young', 'Car=Sports', 'Salary-Level=High'),
        ('Age-Range=Young', 'Car=Sports', 'Salary-Level=Low'),
       ('Age-Range=Middle', 'Car=Sedan', 'Salary-Level=High'),
        ('Age-Range=Young', 'Car=Sports', 'Salary-Level=Low'),
       ('Age-Range=Middle', 'Car=Sedan', 'Salary-Level=High'),
        ('Age-Range=Young', 'Car=Sports', 'Salary-Level=Low')]

        data = pd.DataFrame(dataList, index=indexes, columns=column)
        pd.util.testing.assert_frame_equal(output, data, check_dtype=False)

    def test_maximum_value(self):
        fpi = FPI(dataset_file, 0.3)
        output = fpi.build()
        column = ['Scores']
        indexValue = [('Age-Range=Young', 'Car=Sports', 'Salary-Level=High')]
        expectedValue = 1.9375

        data = pd.DataFrame(expectedValue, index=indexValue, columns=column)
        pd.util.testing.assert_frame_equal(data, output[output['Scores'] == output['Scores'].max()], check_dtype=False)

if __name__ == '__main__':
    unittest.main()








