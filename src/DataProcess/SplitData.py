import pandas as pd
import random

from copy import deepcopy

from sklearn.model_selection import train_test_split

from DataProcess.RawDataPreprocess import process_and_encode_dataframe, set_config, load_df_from_excel

config_path = r"Config\DataConfig.json"
DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

seed = 4
train_size = 0.8
test_size = 1 - train_size

def get_pc_info(df):
  count = df.ProductName.value_counts()
  print(f"Taille du dataframe : {len(df)}")
  print(f"Nombre de codes produit {len(count)}")
  # print(count)

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

def data_aug_codeName(df, DATA_CONFIG):
    """Add the name of the productCode as a description for all productCode in the df

    Args:
        df (pd.DataFrame): The df that is going to be duplicate with productName
        matrix_tree (pd.DataFrame): All name of each class

    Returns:
        df (pd.DataFrame): Same size DataFrame with product names as description only
    """
    matrix_tree_df, _ = load_df_from_excel(DATA_CONFIG["MATRICES_TREE_PATH"], DATA_CONFIG["REQUIERD_COL_TREE"])
    # Keep one occurrence of each product code only
    df = df.drop_duplicates(subset=["ProductCode"])

    # For each productCode add the Name of the ProductCode as a description
    df_add = deepcopy(df)
    # Ensure that matrix_tree has the necessary columns
    if not all(col in matrix_tree_df.columns for col in ["ProductCode", "Nom"]):
        raise ValueError("matrix_tree must contain 'ProductCode' and 'Nom' columns")

    # Merge DataFrames
    df_add = df_add.merge(matrix_tree_df[["ProductCode", "Nom"]], on="ProductCode")
    df_add["Description"] = df_add["Nom"]
    df_add.drop("Nom", inplace=True, axis=1)
    
    # Apply transformation to SampleCode
    if "SampleCode" in df_add.columns:
        df_add["SampleCode"] = df_add["SampleCode"].apply(lambda x: x+'_aug_CN')
    else:
        raise KeyError("Column 'SampleCode' does not exist in the DataFrame.")
    
    return df_add

def swap_aug_data_train_test(df):
    """
    Augmented data are marked with "aug" in the sample code. This function aims to swap all augmented
    data in the test set with non-augmented data with the same SampleCode from the train set.  

    Args:
        df (pd.DataFrame): The combined dataset containing both train and test sets with a "Mode" column.

    Returns:
        df (pd.DataFrame): The updated combined dataset with swapped augmented data.
    """
    
    # Filter the dataset into train and test sets
    df_groups = dict(tuple(df.groupby("Mode")))
    df_train = df_groups.get("train", pd.DataFrame())
    df_test = df_groups.get("test", pd.DataFrame())

    # Filter out the rows in the test set that have augmented data (SampleCode contains "aug").
    df_test_aug = df_test[df_test["SampleCode"].str.contains("aug")]
    swapped_rows = []

    # Iterate over each augmented row in the test set.
    for index, row in df_test_aug.iterrows():
        pc = row["ProductCode"]
        
        # Find non-augmented rows in the train set with the same ProductCode.
        possible_rows = df_train[(df_train["ProductCode"] == pc) & (~df_train["SampleCode"].str.contains("aug"))]
        
        if not possible_rows.empty:
            # Randomly select one of the possible rows.
            v = random.choice(list(possible_rows.index))
            new_row = possible_rows.loc[[v]]
            
            # Append the selected row to the list of swapped rows.
            swapped_rows.append(new_row.values.tolist()[0])
    
    # Create a DataFrame from the swapped rows.
    swapped_df = pd.DataFrame(swapped_rows, columns=df_train.columns)
    
    # Remove the swapped rows from the train set and add the augmented test rows.
    df_train_swapped = df_train[~df_train["SampleCode"].isin(swapped_df["SampleCode"])]
    df_train_swapped = pd.concat([df_train_swapped, df_test_aug])
    
    # Remove the augmented rows from the test set and add the swapped rows.
    df_test_swapped = df_test[~df_test["SampleCode"].isin(df_test_aug["SampleCode"])]
    df_test_swapped = pd.concat([df_test_swapped, swapped_df])

    # Combine the updated train and test sets
    df_combined = pd.concat([df_train_swapped, df_test_swapped])

    return df_combined

def add_augmented_data(df, matrix_tree, drop_duplicates=None, aug_thresh=0):
    # Remove duplicates based on SampleCodes
    df = df.drop_duplicates(subset=["SampleCode"])
    
    # Remove duplicate rows based on specified columns
    if drop_duplicates:
        df = df.drop_duplicates(subset=drop_duplicates)
    
    # Identify and augment low-occurrence product codes
    product_counts = df.ProductCode.value_counts()
    low_occurrence_products = product_counts[product_counts <= aug_thresh].index
    df_low = df[df["ProductCode"].isin(low_occurrence_products)]

    # Augment data for low-occurrence product codes
    df_augmented = data_aug_codeName(df_low, matrix_tree)
    print(f"Added {len(df_augmented)} rows thanks augmentation")

    df_augmented = process_and_encode_dataframe(df_augmented, DATA_CONFIG)
    print("Augmented data are processed and encoded")
    # Combine original and augmented data
    df_combined = pd.concat([df, df_augmented], ignore_index=True)
    
    return df_combined

def split_data(df, test_size=0.2, seed=42):
    """
    Augments low-occurrence products and splits the data into train and test sets,
    adding a 'Mode' column to indicate the split.

    Args:
        df (pd.DataFrame): The input dataframe.
        matrix_tree: Matrix used for data augmentation.
        drop_duplicates (list or None): Columns to use for dropping duplicates. If None, no duplicates are dropped.
        aug_thresh (int): The threshold for augmentation. Default is 5.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        seed (int): Random state for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: The original dataframe with an additional 'Mode' column indicating 'train' or 'test'.
    """

    # Remove products that occur only once after augmentation
    counts = df.ProductCode.value_counts()
    single_occurrence = counts[counts == 1].index
    df = df[~df["ProductCode"].isin(single_occurrence)]
    print(f"Removed {len(single_occurrence)} samples with single occurrence")

    # Perform train-test split
    train_indices, test_indices = train_test_split(
        df.index, test_size=test_size, random_state=seed, 
        stratify=df["ProductCode"]
    )
    df = df.copy()
    # Add 'Mode' column to indicate train/test split
    df['Mode'] = 'train'
    df.loc[test_indices, 'Mode'] = 'test'

    # Swap augmented data in test set with non-augmented data from train set
    df = swap_aug_data_train_test(df)

    # Check for common samples between train and test
    train_samples = set(df[df['Mode'] == 'train']['SampleCode'])
    test_samples = set(df[df['Mode'] == 'test']['SampleCode'])
    common_samples = train_samples.intersection(test_samples)
    print(f"Number of common SampleCodes between train and test: {len(common_samples)}")

    # Display information about the product codes in train and test sets
    for mode in ['train', 'test']:
        print(f"\n{mode.capitalize()} set info:")
        get_pc_info(df[df['Mode'] == mode])

    return df

if __name__ == "__main__":
  print("\n")
  print("-- STARTING THE SPLIT --")

  DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

  matrices_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\matrices_tree.xlsx")

  df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\MAX1000_Merged_and_cleand_df_2024-06-11_0.xlsx")

  droped_col = DATA_CONFIG["FEATURES_COL"]+DATA_CONFIG["TARGET_COL"]
  
  data_train, data_test = split_and_augment_data(df, matrices_df, drop_duplicates=droped_col)
  print("--- The split is done ---")

  print("--- Saving ---")
  data_train.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\1000_train_cleanDescr_image.xlsx", index=False)
  data_test.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\1000_test_cleanDescr_image.xlsx", index=False)
  print("-- END OF THE JOB --")

