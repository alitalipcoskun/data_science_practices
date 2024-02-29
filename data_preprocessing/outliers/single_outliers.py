import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
#It is for importing the dfOps class.
sys.path.append(r"C:\Users\ali_t\data_science\data_science_practices\feature_engineering\data_preprocessing")
from data_preprocessing.operations.dataframe_operations import dfOperations as dfOps


class singleColOutliers:
    """
    
    """

    def __init__(self, 
                 q1: float = 0.10,
                 q3: float = 0.90,
                 categoryTh: int = 8,
                 cardinalTh: int = 20) -> None:
        
        self.setQuartiles(q1, q3)
        self.__categoryTh = categoryTh
        self.__cardinalTh = cardinalTh
    
    def getQuartiles(self) -> list[float]:
        return [self.__q1, self.__q3]
    
    def setQuartiles(self,
                     q1: float = 0.1,
                     q3: float = 0.9) -> None:
        if 0 > q1 > 1 or 0 > q3 > 1 or q1 > q3:
            raise Exception("Quartile values are invalid. Check the values of q1 and q3 again.")
        self.__q1 = q1
        self.__q3 = q3
    

    def getTresholds(self):
        return [self.__cardinalTh, self.__categoryTh]

    def setTresholds(self,
                     cardinalTh: int = 20,
                     categoryTh: int = 8) -> None:
        self.__cardinalTh = cardinalTh
        self.__categoryTh = categoryTh

    def findBounds(self, df: pd.DataFrame, 
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

    def checkOutlier(self, df: pd.DataFrame,
                column: str = "",
                boxplotFlag = False) -> None:

        dfOps.verifyColumn(df, column)
        if boxplotFlag:
            sns.boxplot(x = df[column])
            plt.show()

        [upperBound, lowerBound] = self.findBounds(df, column)
        outlierDf = df[(df[column] < lowerBound) | (df[column] > upperBound)].values
        outlierFlag = outlierDf.any(axis = None)
        if outlierFlag:
            self.outlierInfo(df, column)
        else:
            print(f"{column} has no outlier rows.")

    def seperateColumns(self, df: pd.DataFrame) -> list:
        """
            The function helps you seperate columns as follows: categoric, numeric, ordinal

            Parameters:
                df: It is the dataframe that the columns must be seperated.
                categoricTh: It is a value to assign numeric column but behaves as categoric column
                cardinalTh: It is a valut to assign categoric column but behaves as numeric column

            Returns: [categoric_cols, numeric_cols, ordinal_cols]
                categoric_cols -> list: It holds the categoric typed columns after the function execution
                numeric_cols -> list: It holds the numeric typed columns after the function execution
                ordinal_cols -> list: It holds the ordinal typed columns after the function execution.
        """
        [cardinalTh, categoricTh] = self.getTresholds()
        categoric_cols = [column for column in df.columns if df[column].dtypes == 'O']
        numeric_columns = [column for column in df.columns if df[column].dtypes != 'O']
        num_but_cats = [column for column in df.columns if df[column].nunique() < categoricTh and df[column].dtypes != 'O']

        cat_but_cardinals = [column for column in categoric_cols if df[column].nunique() > cardinalTh]

        categoric_cols = categoric_cols + num_but_cats
        categoric_cols = [column for column in categoric_cols if column not in cat_but_cardinals]

        numeric_columns = [column for column in numeric_columns if column not in num_but_cats]

        return [categoric_cols, numeric_columns, cat_but_cardinals]

    def grabOutlierIndexes(self, df: pd.DataFrame,
                       column: str) -> list:
        [upper_bound, lower_bound] = self.findBounds(df, column)
        indexes = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        return indexes


    def outlierInfo(self, df: pd.DataFrame,
                column: str) -> None:
    
        indexes = self.grabOutlierIndexes(df, column)
        print(f"{column} has {len(indexes)} rows that has outlier value.")
        if len(indexes) > 10:
            print(df.loc[indexes, :].head(), end = "\n\n\n")
        else:
            print(df.loc[indexes, :], end= "\n\n")

    def replaceWithBounds(self, df: pd.DataFrame,
                          column: str) -> pd.DataFrame:
        [upper_bound, lower_bound] = self.findBounds(df, column)
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound

        return df

    def removeOutliers(self, df: pd.DataFrame,
                   column: str) -> pd.DataFrame:
        dfOps.verifyColumn(df, column)
        [upper_bound, lower_bound] = self.findBounds(df, column)

        df = df.loc[~((df[column] < lower_bound) | (df[column] > upper_bound)), :]
        return df

