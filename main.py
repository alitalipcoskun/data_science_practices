import sys
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering")
from data_preprocessing.outliers import outlier_handler
from data_preprocessing.df_operations import dfOps
from data_preprocessing.nan_operations import nan_handler

if __name__ == '__main__':
    outlierHandler = outlier_handler()
    df_operations = dfOps(path= r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering\datasets\titanic.csv")
    df = df_operations.loadCsvDataset()
    df = df_operations.lowercaseColNames(df)
    [categoric, numeric, cardinal] = df_operations.seperateColumns(df) 
    
    print(categoric)
    print(cardinal)
    numeric = [column for column in numeric if not column.__contains__("id")]
    print(numeric)
    nan_handle = nan_handler()
    missing_df = nan_handle.nan_information(df)
    print(missing_df.to_string(index= False))
    print(missing_dff)
    