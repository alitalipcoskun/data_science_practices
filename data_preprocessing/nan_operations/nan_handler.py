import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import sys
#It is for importing the dfOps class.
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering\data_preprocessing")
from df_operations import dfOps


class nan_handler:
    def __init__(self) -> None:
        pass


    def find_nan_columns(self, df: pd.DataFrame) -> None:
        """
        It finds the columns that has null value in at least one row.

        args:
            df -> pd.DataFrame: dataframe that wanted to be checked whether
            it has null value on its rows or not.
        
        returns:
            nan_columns -> list[str]: The list contains columns that has one null
            value on its rows.
        """
        nan_columns = [column for column in df.columns if df[column].isnull().sum() > 0]
        
        return nan_columns
    
    def nan_information(self, 
                        df: pd.DataFrame
                        ) -> pd.DataFrame:
        
        nan_columns = self.find_nan_columns(df)

        n_miss = []
        ratio = []
        for column in nan_columns:
            n_miss.append(df[column].isnull().values.sum())
            curr_ratio = np.round(n_miss[-1] / df.shape[0] * 100, 2)
            ratio.append(curr_ratio)
        
        output_df = pd.DataFrame({
            "Columns": nan_columns,
            'n_miss': n_miss,
            "Ratio": ratio 
        })
        output_df = output_df.sort_values(by = ['n_miss', 'Ratio'], ascending= False)

        return output_df
    
    def fill_with_mean(self,
                       df: pd.DataFrame) -> pd.DataFrame:
        """
        It fills the numeric columns of dataframe with mean values. It does not
        touch or fill other type of columns.

        args:
            df -> pd.DataFrame: dataframe that has null numeric rows and wanted to be
            filled with respect to mean value of column.

        returns:
            output_df -> pd.DataFrame: dataframe that filled with respect to the function
            definition. 
        
        """    
        output_df = df.copy()
        output_df = output_df.apply(lambda column: column.fillna(column.mean()) if(column.dtype != "O") else column)

        return output_df
    
    def fill_with_mode(self,
                       df: pd.DataFrame,
                       cardinalTh: int = 20) -> pd.DataFrame:

        output_df = df.copy()
        output_df = output_df.appy(lambda column: column.fillna(column.mode()) if(column.dtype == 'O'and len(column.unique())  <=  cardinalTh) else column)    
        return output_df
    
    
    def fill_with_cat_mean(self,
                      df: pd.DataFrame,
                      filled_col: str,
                      group_col: str) -> pd.DataFrame:
        """
        The function first groups the dataframe by with respect to the group_col 
        argument and finds the mean value of filled_col argument. Finally, it fills
        the NaN values of the rows of filled_col argument.

        args:
            df: dataframe that wanted to be filled.
            filled_col: the column that targeted to be filled.
            group_col: the column that the dataframe wanted to be grouped.

        returns:
            output_df -> pd.DataFrame: it is the dataframe that
            filled filled_col arguments by mean with respect to the group by 
            group_col argument.  
        
        """
        checker = dfOps()
        checker.verifyColumn(df, filled_col)
        checker.verifyColumn(df, filled_col)
        #Copying is for protecting the original one in function.
        output_df = df.copy()
        output_df[filled_col] = df[filled_col].fillna(df.groupby(group_col)[filled_col].transform("mean"))

        return output_df

    def knn_imputation(self, 
                       df: pd.DataFrame,
                       n_neighbors: int = 5) -> pd.DataFrame:
        
        """
        It is a technique for filling in missing values by estimating them based on the characteristics of
        similar neighbouring data points.

        The approach is suitable for "missing completely at random" (MCAR) or (MAR).
        
        args:
            df -> pd.DataFrame: dataframe that wanted to be imputated.
            n_neighbors: int = 5: It is a parameter to be used in KNNImputer. It helps you to visit the amount of parameter neighbors to
            fill the NaN columns. It is assigned to 5 by default.
        returns:
            output_df -> pd.DataFrame: dataframe that has no NaN values. It is filled with respect
            to the sklearn.impute.KNNImputer.
        """
        assert n_neighbors > 1, f"{n_neighbors} is not a valid to be used in KNNImputer. Try to give an integer that is bigger than 1."
        imputer = KNNImputer(n_neighbors= n_neighbors)
        output_df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

        return output_df


