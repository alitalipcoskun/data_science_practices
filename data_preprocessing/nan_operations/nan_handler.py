import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class nan_handler:
    def __init__(self) -> None:
        pass


    def find_nan_columns(self, df: pd.DataFrame) -> None:
        
        nan_columns =[column for column in df.columns if df[column].isnull().sum() > 0]
        
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
    
    def fill_with_cat_mean(self,
                      df: pd.DataFrame,
                      filled_col: str,
                      group_col: str) -> pd.DataFrame:
        
        df[filled_col] = df[filled_col].fillna(df.groupby(group_col)[filled_col].transform("mean"))

        return df



