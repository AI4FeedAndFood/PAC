import pandas as pd
import json
import os
import random

from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from RawDataPreprocess import clean_description, set_config

config_path = r"Config\DataConfig.json"
DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

seed = 4
train_size = 0.8
test_size = 1 - train_size

def get_pc_info(df):
  count = df.ProductName.value_counts()
  print(f"Taille du dataframe : {len(df)}")
  print(f"Nombre de codes produit {len(count)}")
  print(count)

def extract_duplicates(df):
    """
    Identifies and returns rows in the dataframe that have duplicated 
    'CleanDescription' and 'ProductCode' values.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: A dataframe containing only the duplicated rows.
    """
    mask = df.duplicated(subset=['CleanDescription', 'ProductCode'], keep=False)
    df = df[mask].sort_values("CleanDescription")
    print(df[['CleanDescription', 'ProductCode']].head(20))
    return df[mask]

def data_aug_codeName(df, matrix_tree):
  """Add the name of the productCode as a description for all productCode in the df

  Args:
      df (pd.DataFrame): The df that is going to be duplicate with productName
      matrix_tree (pd.DataFrame): All name of each class

  Returns:
      df (pd.DataFrame): Same size DataFrame with product names as description only
  """
  # Keep one occurence of each product code only
  df = df.drop_duplicates(subset=["ProductCode"])

  # For each productCode add the Name of the ProductCode as a description
  df_add = deepcopy(df)
  df_add = df_add.merge(matrix_tree[["ProductCode", "Nom"]], on="ProductCode")
  df_add["Description"] = df_add["Nom"]
  df_add.drop("Nom", inplace=True, axis=1)

  # Clean the name according to the raw data processong
  df_add = clean_description(df_add, DATA_CONFIG["LANG"], DATA_CONFIG["DEL_LIST"], DATA_CONFIG["CONVERT_DICT"])
  df_add["SampleCode"] = df_add["SampleCode"].apply(lambda x: x+'_aug_CN')
  
  return df_add

def swap_augData_train_test(df_train, df_test):
    """
    Augmented data are marked with "aug" in the sample code. This function aims to swap all augmented
    data in the test set with non-augmented data with the same SampleCode from the train set.  

    Args:
        df_train (pd.DataFrame): The train set.
        df_test (pd.DataFrame): The test set.

    Returns:
        df_train (pd.DataFrame): The updated train set.
        df_test (pd.DataFrame): The updated test set.
    """
    
    # Filter out the rows in the test set that have augmented data (SampleCode contains "aug").
    df_test_augg = df_test[df_test["SampleCode"].str.contains("aug")]
    swaped_rows = []

    # Iterate over each augmented row in the test set.
    for index, row in df_test_augg.iterrows():
        pc = row["ProductCode"]
        
        # Find non-augmented rows in the train set with the same ProductCode.
        possible_row = df_train[df_train["ProductCode"] == pc]
        possible_row = possible_row[~possible_row["SampleCode"].str.contains("aug")]
        
        # Randomly select one of the possible rows.
        v = random.choice(list(possible_row.index))
        new_row = possible_row.loc[[v]]
        
        # Append the selected row to the list of swapped rows.
        swaped_rows.append(new_row.values.tolist()[0])
    
    # Create a DataFrame from the swapped rows.
    swaped_df = pd.DataFrame(swaped_rows, columns=df_train.columns)
    
    # Remove the swapped rows from the train set and add the augmented test rows.
    df_train_swaped = df_train[~df_train["SampleCode"].isin(swaped_df["SampleCode"])]
    df_train_swaped = pd.concat([df_train_swaped, df_test_augg])
    
    # Save the augmented test set and the swapped DataFrame to Excel files.
    df_test_augg.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\df_test_augg.xlsx")
    swaped_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\swaped_df.xlsx")
    
    # Remove the augmented rows from the test set and add the swapped rows.
    df_test_swaped = df_test[~df_test["SampleCode"].isin(df_test_augg["SampleCode"])]
    df_test_swaped = pd.concat([df_test_swaped, swaped_df])

    return df_train_swaped, df_test_swaped

def train_test_split_with_aug(df, matrix_tree, drop_duplicates ,aug_thresh=5):
    """
    This function performs a train-test split on the input dataframe after handling
    products with low occurrence by augmenting them. It ensures that products with 
    low occurrence (less than or equal to aug_thresh) are augmented before splitting 
    the data.

    Args:
        df (pd.DataFrame): The input dataframe.
        matrix_tree: Matrix used for data augmentation.
        aug_thresh (int): The threshold for augmentation, default is 5.

    Returns:
        pd.DataFrame: The training set.
        pd.DataFrame: The test set.
    """
    
    # Remove duplicate rows based on 'CleanDescription'.
    if drop_duplicates:
        df_unique = df.drop_duplicates(subset=drop_duplicates)
    else:
       df_unique = df
    
    # Handle product codes that occur fewer than or equal to aug_thresh times.
    count = df_unique.ProductCode.value_counts()
    count_thresh = count[count <= aug_thresh]
    df_low = df_unique[df_unique["ProductCode"].isin(count_thresh.index)]

    # Augment data for low-occurrence product codes.
    df_add_CN = data_aug_codeName(df_low, matrix_tree)
    print(f"{len(df_add_CN)} added rows thanks to augmentation")

    # Concatenate the original dataframe with the augmented data.
    df = pd.concat([df, df_add_CN])
    count = df.ProductCode.value_counts()
    
    # Identify product codes that still occur only once.
    c1 = count[count <= 1]
    print("Check for the split, unique codes:", len(df[df["ProductCode"].isin(c1.index)]))
    
    # Remove rows with product codes that occur only once.
    if len(df[df["ProductCode"].isin(c1.index)]) > 0:
        print("Removing:", df[df["ProductCode"].isin(c1.index)])
        df = df[~df["ProductCode"].isin(c1.index)]

    # Perform the train-test split.
    data_train, data_test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["ProductCode"])

    # Swap augmented data in the test set with non-augmented data from the train set.
    data_train, data_test = swap_augData_train_test(data_train, data_test)

    # Display information about the product codes in the training set.
    print("\nTrain infos")
    get_pc_info(data_train)

    # Display information about the product codes in the test set.
    print("\nTest infos")
    get_pc_info(data_test)

    return data_train, data_test

if __name__ == "__main__":
  print("\n")
  print("-- STARTING THE SPLIT --")

  DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

  matrices_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\matrices_tree.xlsx")

  df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\cleanedDescr_Image_df_2024-06-11_0.xlsx")
  
  data_train, data_test = train_test_split_with_aug(df, matrices_df, drop_duplicates=DATA_CONFIG["FEATURES_COL"]+DATA_CONFIG["TARGET_COL"])
  print("The split is done")

  print("Saving")
  data_train.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\train_cleanDescr_image.xlsx", index=False)
  data_test.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\test_cleanDescr_image.xlsx", index=False)
  print("-- END OF THE JOB --")

