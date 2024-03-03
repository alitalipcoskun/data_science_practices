import sys
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering")
from data_preprocessing.outliers import outlier_handler
from data_preprocessing.df_operations import dfOps
from data_preprocessing.nan_operations import nan_handler

if __name__ == '__main__':
    outlierHandler = outlier_handler()
    df_operations = dfOps()
    df = df_operations.loadCsvDataset(path= r"datasets\titanic.csv")
    df = df_operations.lowercaseColNames(df)
    [categoric, numeric, cardinal] = df_operations.seperateColumns(df) 
    
    print(categoric)
    print(cardinal)
    numeric = [column for column in numeric if not column.__contains__("id")]
    print(numeric)
    nan_handle = nan_handler()
    missing_df = nan_handle.nan_information(df)
    print(missing_df.to_string(index= False))
    #dff = df_operations.one_hot_encoder(df= df, cat_cols= categoric, num_cols= numeric)
    dff = nan_handle.fill_with_mean(df)
    filled_dff = nan_handle.fill_with_cat_mean(dff, "age", "sex")
    missing_dff = nan_handle.nan_information(filled_dff)
    print(missing_dff)
    