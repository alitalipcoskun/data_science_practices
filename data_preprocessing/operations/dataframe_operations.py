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