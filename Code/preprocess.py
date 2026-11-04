"""
The code used in order to preprocess all the data
"""
import pandas as pd

def remove_no_consent(df,col,remove_nan):
    """
    input:
    df : Pandas dataframe 
    col: The consent column of the df 
    remove_nan: Removing NaN values or not

    output:

    df : Pandas dataframe with only questioners who gave consent and possibly no NaN values.

    """
    # Copy original data frame to compare to new one
    df_before = df.copy()

    # Keep people who gave consents (A)
    df = df[col=='A']

    if remove_nan:
        df = df.dropna()

    print("Shape:",df.shape)
    print("Number of people who gave consent:",df.shape[0])
    print(f"Difference with total: {df_before.shape[0] - df.shape[0]}, Percentage: {((df.shape[0] / df_before.shape[0]) *100):.2f}%")

    return df

def translate(df):
    """
    Input:
    df : Pandas Dataframe
    
    Output:

    df: Translate the columns in the right manner for one hot encoding + normalization
    """
    # Question 53 and 54 will be translated to numerical values
    df["53"] = df["53"].replace(["A","B","C","D","E",],[1,2,3,4,5])
    df["54"] = df["54"].replace(["A","B","C","D","E",],[1,2,3,4,5])
    # Question 52 needs to be represented as 6 distinct columns to indicate if strategy was present (0=no 1=yes)
    # print("\nDifferent occurences question 52:",len(df["52"].unique()))

    answers_52 = ["A","B","C","D","E","F"]

    for answer in answers_52:
        
        # Ones if the choice (A - F) is present in the answer
        ones = [1  if answer in x else 0 for x in df["52"]]
        df[f"52.{answer}"] = ones

    # print("Before",df.head())

    # Drop old column
    df = df.drop('52', axis=1)
    # Re order columns
    cols = df.columns.tolist()
    cols = cols[3:] + cols[1:3] # first columns 52.answer,53,54
    df = df[cols]
    df_unnorm = df.copy()
    # column 53 and 54 need normalization (values between 0 and 1 instead of 1 till 5).
    def normalize_0_1(col):
        return (col-1)/4
    df['53'] = normalize_0_1(df['53'])
    df['54'] = normalize_0_1(df['54'])
    # print("\nafter\n",df.head())
    return df
