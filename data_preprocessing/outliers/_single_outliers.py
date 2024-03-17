import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
#It is for importing the dfOps class.
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering\data_preprocessing")
from df_operations.dataframe_operations import df_operations as dfOps


class singleColOutliers:
    
    def __init__(self,
                 path: str, 
                 q1: float = 0.10,
                 q3: float = 0.90) -> None:
        
        self.__df_path = path
        df_path = self.get_df_path()
        checker = dfOps(path= df_path)
        self.__checker = checker
        self.setQuartiles(q1, q3)


    def set_df_path(self,
                    path: str) -> None:
        
        self.__df_path = path
    

    def get_df_path(self) -> str:

        return self.__df_path
    
    def get_column_checker(self)-> dfOps:

        return self.__checker

    def set_column_checker(self,
                        checker: dfOps) -> None:
        self.__checker = checker
    
    def getQuartiles(self) -> list[float]:
        return [self.__q1, self.__q3]
    
    def setQuartiles(self,
                    q1: float = 0.1,
                    q3: float = 0.9) -> None:
        if 0 > q1 > 1 or 0 > q3 > 1 or q1 > q3:
            raise Exception("Quartile values are invalid. Check the values of q1 and q3 again.")
        self.__q1 = q1
        self.__q3 = q3
    



    def findBounds(self,
                df: pd.DataFrame, 
                column: str) -> list:
        """
        This function is created to be used as an inner function.
        It finds the upper and the lower bounds with respect to the
        q1 and the q3 values.
        iqr = (inter quartile range)
        Parameters:
            df: It is the dataframe to be used for finding bounds for specified column
            column: It is string that must be in the list of the dataframe columns.
            q1: It is used to find out the data under that percentage.
        Returns: list
            [upperBound, lowerBound]
            upperBound -> float: It is the maximum value that to be counted as normal value. If a value is bigger than upperBound, then it is an outlier.
            lowerBound -> float: It is the minimum value that to be counted as normal value. If a value is less than lowerBound, then it is an outlier.

        """
        [q1, q3] = self.getQuartiles()
        #It finds out the data under of the (q1*100)% for specified column 
        quarter1 = df[column].quantile(q1)

        #It finds out the data under of the (q3*100)% for specified column
        quarter3 = df[column].quantile(q3)

        iqr = quarter3 - quarter1

        lowerBound = quarter1 - 1.5 * iqr

        upperBound = quarter3 + 1.5 * iqr

        return [upperBound, lowerBound]

    def checkOutlier(self, 
                    df: pd.DataFrame,
                    column: str = "",
                    boxplotFlag = False) -> None:
        """
        The function does give info about the given column for outlier values.
        It finds out upper and lower bounds, with respect to the given q1 and q3 values in the class.
        Then checks whether column has null value or not.
        
        args:
            df -> pd.DataFrame: the dataframe that wanted to be analyzed
            column -> str: the column that wanted to be analyzed.
            boxplotFlag -> bool: the parameter for seeing the boxplot while making analysis.
        
        returns:
            None
        """


        checker = self.get_column_checker()
        checker.verifyColumn(df, column)
        #Prints boxplot
        if boxplotFlag:
            sns.boxplot(x = df[column])
            plt.show()

        #Finds upper and the lower bound for finding out the outlier values.
        [upperBound, lowerBound] = self.findBounds(df, column)
        outlierDf = df[(df[column] < lowerBound) | (df[column] > upperBound)].values
        outlierFlag = outlierDf.any(axis = None)
        if outlierFlag:
            self.outlierInfo(df, column)
        else:
            print(f"{column} has no outlier rows.")



    def grabOutlierIndexes(self,
                        df: pd.DataFrame,
                        column: str) -> list:
        """
            The function finds out the indexes of outlier values for given column.

            args:
                df -> pd.DataFrame: the dataframe that wanted to be analyzed.
                column -> str: the string that is column of dataframe and wanted to get reach to the indexes of outlier values.
            
            returns:
                indexes -> list[int]: returns the list of the indexes of outlier values for given column.
        """

        #Finding out upper and the lower bound to reach out the indexes of outlier values.
        [upper_bound, lower_bound] = self.findBounds(df, column)
        indexes = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        return indexes


    def outlierInfo(self, df: pd.DataFrame,
                column: str) -> None:

        checker = self.get_column_checker()
        checker.verifyColumn(df, column)
        indexes = self.grabOutlierIndexes(df, column)
        print(f"{column} has {len(indexes)} rows that has outlier value.")
        if len(indexes) > 10:
            print(df.loc[indexes, :].head(), end = "\n\n\n")
        else:
            print(df.loc[indexes, :], end= "\n\n")

    def replaceWithBounds(self, df: pd.DataFrame,
                        column: str) -> pd.DataFrame:
        
        checker = self.get_column_checker()
        checker.verifyColumn(df, column)
        [upper_bound, lower_bound] = self.findBounds(df, column)
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound

        return df

    def removeOutliers(self, df: pd.DataFrame,
                    column: str) -> pd.DataFrame:
        
        checker = self.get_column_checker()
        checker.verifyColumn(df, column)
        [upper_bound, lower_bound] = self.findBounds(df, column)

        df = df.loc[~((df[column] < lower_bound) | (df[column] > upper_bound)), :]
        return df

