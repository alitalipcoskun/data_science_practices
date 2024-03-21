import sys
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering")
from data_preprocessing.outliers import outlier_handler
from data_preprocessing.df_operations import dfOps
from data_preprocessing.nan_operations import nan_handler

if __name__ == '__main__':
    path= r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering\datasets\titanic.csv"
    target = 'survived'
    outlierHandler = outlier_handler(path= path)
    df_operations = dfOps(path = path)
    df = df_operations.loadCsvDataset()
    df = df_operations.lowercaseColNames(df)
    [categoric, numeric, cardinal] = df_operations.seperateColumns(df)
    categoric= [column for column in df.columns if column != target] 
    numeric = [column for column in numeric if not column.__contains__("id")]
    nan_handle = nan_handler()
    missing_df = nan_handle.nan_information(df)
    print(missing_df)
    #output_df = df_operations.category_convert(df, categoric_cols=categoric)
    #output_df = df_operations.rare_encoder(output_df, cat_cols= categoric, rare_th= 0.03)
    #df_operations.rare_analyser(df = output_df, cat_cols= categoric, target_col= target)
    print(numeric)
    print(categoric)
    print(cardinal)
    print(df['fare'].value_counts())