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
        df = df.dropna(subset=['43','44','45'])

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
    # Question 44 and 45 will be translated to numerical values
    df["44"] = df["44"].replace(["A","B","C","D","E"],[1,2,3,4,5])
    df["45"] = df["45"].replace(["A","B","C","D","E"],[1,2,3,4,5])
    # Question 43 needs to be represented as 10 distinct columns to indicate if strategy was present (0=no 1=yes)
    

    answers_43 = ["A","B","C","D","E","F","G","H","I","J"]

    
    for answer in answers_43:
        
        # Ones if the choice (A - J) is present in the answer
        ones = [1  if isinstance(x,str) and answer in x else 0 for x in df["43"]] # Some people did not fill this one in -> NaN value 
        df[f"43.{answer}"] = ones

    # print("Before",df.head())

    # Drop old column
    df = df.drop('43', axis=1)
    # Re order columns
    cols = df.columns.tolist()
    
    cols = cols[7:] + [cols[0]] + cols[2:7]  # first columns 43.answer + exclude consent column

    df = df[cols]

    df_unnorm = df.copy()
    # column 44 and 45 need normalization (values between 0 and 1 instead of 1 till 5).
    def normalize_0_1(col):
        return (col-1)/4
    df['44'] = normalize_0_1(df['44'])
    df['45'] = normalize_0_1(df['45'])
    # print("\nafter\n",df.head())
    return df,df_unnorm
