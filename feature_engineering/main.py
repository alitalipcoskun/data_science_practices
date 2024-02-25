import sys
from data_preprocessing.dataframe_operations import dfOperations as dfOps
from data_preprocessing.handle_outliers import handleOutliers as handler




if __name__ == '__main__':
    outlierHandler = handler()
    df = dfOps.loadCsvDataset(path= r"datasets\titanic.csv")
    df = dfOps.lowercaseColNames(df)
    [categoric, numeric, cardinal] = outlierHandler.seperateColumns(df) 
    print(numeric)
    for column in numeric:
        df = outlierHandler.removeOutliers(df, column)
        outlierHandler.checkOutlier(df, column)
