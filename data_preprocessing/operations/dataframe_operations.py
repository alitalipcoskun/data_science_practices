import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class dfOperations:
    def loadCsvDataset(path:str = "datasets\diabetes.csv") -> pd.DataFrame:
        """
        Takes the path of the dataset, which is csv file, returns the dataframe.
            path: Takes a string path argument to return the dataframe.

        Returns:
            df: Returns the dataframe that is located to the path.

        """
        df = pd.read_csv(path)
        return df

    def lowercaseColNames(df: pd.DataFrame) -> pd.DataFrame:
        """
        The function takes a dataframe as input and returns the dataframe as same. However, it changes the column names to lowercased version of the original.

        Parameters:
            df -> pd.DataFrame: It is the dataframe that the name of the columns are converted to lowercase.

        Returns: pd.DataFrame
            df -> pd.DataFrame: It is the same dataframe as input value but its column names are converted to the lowercase.

        """
        df.columns = [column.lower() for column in df.columns]
        return df

    def verifyColumn(df: pd.DataFrame,column: str):
        """
        The function finds out whether the input string is in the list of dataframe's columns or not

            Parameters:
                df: The dataframe that must be checked to find out whether the input string is column of its or not.
                column: The string that to gets checked in this function.

            Returns:
                None
        """
        if column not in df.columns:
            raise Exception(f"{column} is not in the list of dataframe columns.")
    def seperateColumns(df: pd.DataFrame,
                        categoricTh: int = 8,
                        cardinalTh: int = 20) -> list:
        """
            The function helps you seperate columns as follows: categoric, numeric, ordinal

            Parameters:
                df: It is the dataframe that the columns must be seperated.
                categoricTh: It is the treshold for categoric variables. It is defined 8 as default.
                cardinalTh: It is the treshold for cardinal variables. It is defined 20 as default.

            Returns: [categoric_cols, numeric_cols, ordinal_cols]
                categoric_cols -> list: It holds the categoric typed columns after the function execution
                numeric_cols -> list: It holds the numeric typed columns after the function execution
                ordinal_cols -> list: It holds the ordinal typed columns after the function execution.
        """
        categoric_cols = [column for column in df.columns if df[column].dtypes == 'O']
        numeric_columns = [column for column in df.columns if df[column].dtypes != 'O']
        num_but_cats = [column for column in df.columns if df[column].nunique() < categoricTh and df[column].dtypes != 'O']

        cat_but_cardinals = [column for column in categoric_cols if df[column].nunique() > cardinalTh]

        categoric_cols = categoric_cols + num_but_cats
        categoric_cols = [column for column in categoric_cols if column not in cat_but_cardinals]

        numeric_columns = [column for column in numeric_columns if column not in num_but_cats]

        return [categoric_cols, numeric_columns, cat_but_cardinals]   