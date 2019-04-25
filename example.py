from FPI import FPI
import pandas as pd

data = pd.read_csv("test/data/trainData.csv", sep=";")
labels = pd.read_csv("test/data/trainDataLabels.csv", sep=";")

fpi = FPI(data, 0.3)

fpiFit = FPI.fit(fpi)
fpiPredict = FPI.predict(data, fpiFit, 50)

results = fpiPredict.join(labels)

print(accuracy_score(results["Class"], results["Predicted"]))
print(results['Predicted'].value_counts())
print(results['Class'].value_counts())

cm = confusion_matrix(results['Class'], results['Predicted'])
print(cm)

exp_series = pd.Series(results['Class'])
pred_series = pd.Series(results['Predicted'])
cross = pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)

print(cross)