import sys
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering")
from data_preprocessing.outliers import *
from data_preprocessing.operations import *

if __name__ == '__main__':
    outlierHandler = outlierHandler()
    df = dfOps.loadCsvDataset(path= r"datasets\titanic.csv")
    df = dfOps.lowercaseColNames(df)
    [categoric, numeric, cardinal] = outlierHandler.seperateColumns(df) 
    print(numeric)
    for column in numeric:
        df = outlierHandler.removeOutliers(df, column)
        outlierHandler.checkOutlier(df, column)
