import sys
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering")
from data_preprocessing.outliers import outlier_handler
from data_preprocessing.df_operations import dfOps
from data_preprocessing.nan_operations import nan_handler

if __name__ == '__main__':
    outlierHandler = outlier_handler()
    df = dfOps.loadCsvDataset(path= r"datasets\titanic.csv")
    df = dfOps.lowercaseColNames(df)
    [categoric, numeric, cardinal] = dfOps.seperateColumns(df) 
    
    print(categoric)
    print(cardinal)
    numeric = [column for column in numeric if not column.__contains__("id")]
    print(numeric)
    nan_handle = nan_handler()
    missing_df = nan_handle.nan_information(df)
    print(missing_df.to_string(index= False))

    