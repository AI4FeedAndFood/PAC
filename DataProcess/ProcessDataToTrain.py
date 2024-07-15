import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def load_train_test_data(train_path, drop_na_cols, drop_duplicates_cols=[], test_path=None, split_ratio=0.8, seed=42):
    """
    Load and preprocess training and testing data.

    Args:
        train_path (str): Path to the training data file.
        used_columns (list): List of columns to be used in the analysis.
        test_path (str, optional): Path to the test data file. If None, test data will be split from train data.
        split_ratio (float): Ratio of data to be used for training when splitting.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Preprocessed training and testing DataFrames.
    """
    # Load training data
    train_base = pd.read_excel(train_path)

    # Split data or load separate test data
    if test_path is None:
        train_base, test_base = train_test_split(train_base, train_size=split_ratio, 
                                                 random_state=seed, stratify=train_base["ProductCode"])
    else:
        test_base = pd.read_excel(test_path)


    # Remove rows with NaN values in used columns
    train_base = train_base.dropna(subset=drop_na_cols)
    test_base = test_base.dropna(subset=drop_na_cols)

    # Drop duplicates for the subset
    if drop_duplicates_cols:
        train_base = train_base.drop_duplicates(subset=drop_duplicates_cols)
        test_base = test_base.drop_duplicates(subset=drop_duplicates_cols)

    # Print number of null rows for each used column
    for dataset, name in [(train_base, "Train"), (test_base, "Test")]:
        print(f"\n{name} dataset:")
        for col in drop_na_cols:
            null_count = dataset[col].isnull().sum()
            print(f"{col}: {null_count} null rows")

    return train_base, test_base

def filter_productcode(train, test, min_samples=50, max_samples=None, seed=42, split_ratio=0.8):
    """
    Filter ProductCode based on minimum and maximum number of samples.

    Args:
        train (DataFrame): Training data.
        test (DataFrame): Testing data.
        min_samples (int): Minimum number of samples required for a ProductCode.
        max_samples (int, optional): Maximum number of samples allowed for a ProductCode.
        seed (int): Random seed for reproducibility.
        split_ratio (float): Ratio used for train-test split.

    Returns:
        tuple: Filtered training and testing DataFrames.
    """
    train_samples = set(train['SampleCode'])
    test_samples = set(test['SampleCode'])

    # Trouver les SampleCodes en commun
    common_samples = train_samples.intersection(test_samples)
    # Afficher le résultat
    print(f"Nombre de SampleCodes en commun : {len(common_samples)}")

    def _limit_max_samples(df, product_codes, max_samples):
        filtered_df = df[~df["ProductCode"].isin(product_codes.index)]
        for code in product_codes.index:
            code_df = df[df["ProductCode"] == code]
            if len(code_df) >= max_samples:
                filtered_df = pd.concat([filtered_df, code_df.sample(n=max_samples, random_state=seed)])
        return filtered_df

    # Filter based on minimum samples
    product_code_counts = train.ProductCode.value_counts()
    valid_product_codes = product_code_counts[product_code_counts >= min_samples].index
    train = train[train["ProductCode"].isin(valid_product_codes)]
    test = test[test["ProductCode"].isin(valid_product_codes)]

    # Limit maximum samples if specified
    if max_samples is not None:
        train_product_code_counts = train.ProductCode.value_counts()
        train = _limit_max_samples(train, train_product_code_counts[train_product_code_counts >= max_samples], max_samples)
        
        test_max_samples = int(max_samples * (1 - split_ratio))
        test_product_code_counts = test.ProductCode.value_counts()
        test = _limit_max_samples(test, test_product_code_counts[test_product_code_counts >= test_max_samples], test_max_samples)

    print(f"Train contains {len(train)} rows")
    print(f"Test contains {len(test)} rows")

    # Verify that all test ProductCodes are in train
    train_product_codes = set(train.ProductCode.unique())
    test_product_codes = set(test.ProductCode.unique())
    if test_product_codes.issubset(train_product_codes):
        print("All test ProductCodes are present in train data")
    else:
        print("Warning: Some test ProductCodes are not in train data")

    return train, test

def get_class_distribution(df, matrix_tree_df):
    """
    Analyze and visualize the distribution of ProductCodes.

    Args:
        df (DataFrame): Data to analyze.

    Returns:
        Series: ProductCode value counts.
    """
    product_code_counts = df.ProductCode.value_counts()
    print(f"There are {len(product_code_counts)} unique ProductCodes")
    
    plt.figure(figsize=(20, 10))
    product_code_counts.plot(kind='bar')
    plt.title("Distribution of ProductCodes")
    plt.xlabel("ProductCode")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return product_code_counts

def change_by_matrix_path(df, matrix_tree_df):
    """
    Replace 'ProductCode' in df with 'Matrix Ancestries' from matrix_tree_df if a match is found.
    'ProductCode' remains unchanged if it contains a dot ('.').

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'ProductCode' column to be modified.
    matrix_tree_df (pd.DataFrame): DataFrame containing 'ProductCode' and 'Matrix Ancestries' columns 
                                   for mapping replacement values.

    Returns:
    pd.DataFrame: DataFrame with 'ProductCode' modified according to the mapping.
    """

    def replace_code(product_code):
        # If 'ProductCode' contains a dot, return it unchanged
        if "." in product_code:
            return product_code
        # Find matching rows in matrix_tree_df
        matching_rows = matrix_tree_df[matrix_tree_df["ProductCode"] == product_code]
        # If a match is found, return the 'Matrix Ancestries' value, otherwise return the original 'ProductCode'
        if not matching_rows.empty:
            return matching_rows["Matrix Ancestries"].values[0]
        else:
            return product_code

    # Apply the replace_code function to each element in the 'ProductCode' column
    df["ProductCode"] = df["ProductCode"].apply(replace_code)

    return df

def move_files_based_on_df(df, source_folder, destination_folder):
    """
    Moves files from the source folder to the destination folder if their names are listed in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing an "ImageName" column with the names of the files to move.
    source_folder (str): The directory to move files from.
    destination_folder (str): The directory to move files to.
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate over each image name in the DataFrame
    l = len(df["ImageName"])
    for i, image_name in enumerate(df["ImageName"]):
        if i % 500 == 0:
            print(f"Iteration {i}/{l}")
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        try:
            shutil.move(source_path, destination_path)
        except:
            pass

if __name__ == "__main__":

    # train_path = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\train_cleanDescr_image.xlsx"
    # test_path = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\test_cleanDescr_image.xlsx"

    # used_columns = ["CleanDescription", "AccountCode", "Laboratory", "ProductCode", "ImageName"]

    # train_base, test_base = load_train_test_data(train_path, used_columns, test_path=test_path)

    # train, test = filter_productcode(train_base, test_base, min_samples=100, max_samples=200, seed=42, split_ratio=0.8)

    # train.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\rush_train_cleanDescr_image.xlsx")
    # test.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\rush_test_cleanDescr_image.xlsx")

    train = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\rush_train_cleanDescr_image.xlsx")
    test = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\rush_test_cleanDescr_image.xlsx")

    source_folder =  r"C:\Users\CF6P\Desktop\PAC\Data_PAC\scans"
    destination_folder = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\rush_scans"

    move_files_based_on_df(train, source_folder, destination_folder)
    move_files_based_on_df(test, source_folder, destination_folder)
