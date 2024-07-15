
"""

This script aim to process a raw EXCEL FILE to get an available dataframe to put in the model.

All given files should be in excel format (not .cvs) and should have these columns with the exact spelling.

Required columns to process descrpition and client only (EOL case):

- AccountCode : The code of the client who send the sample
- Description : The description of the sample
- ProductCode : The target variable, the MATRICES assigned to the sample

Required column to process the image of the sample to:

- SampleCode : The code of the sample used to link the row and the image of the sample

Useful columns for interpretation or comprehension:

- ProductName : The full name of the productCode
- ClientType : Interco or client, could help for undertanding or debugging

"""

import json
import pandas as pd
import os
import re

from datetime import date
import numpy as np

from copy import deepcopy
from unidecode import unidecode

from num2words import num2words

MAX_EXCEL_ROW = 1048574

def get_all_tables(path, ext_list=[".csv", ".xlsx"]):
    """Extract all excel from a folder

    Args:
        path (_type_): _description_

    """
    # all path in folder
    docs = os.listdir(path)

    # Complete path
    docs = [os.path.join(path, doc) for doc in docs]

    table_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() in ext_list]

    return table_in_folder

def set_config(config_path, version="VERSION_NG_FR"):
    """Extract conig from the config path

    Args:
        config_path (string path)
        version (str, optional): The laboratory where the tool is going to be used. Defaults to "VERSION_NG_FR".

    Returns:
        DATA_CONFIG (dict)
    """

    DATA_CONFIG = json.load(open(config_path,))[version]

    return DATA_CONFIG

def load_df_from_excel(table_path, REQUIERD_COL):
    """Return a pandas DataFrame from an .xlsx path

    Args:
        excel_path (str): path to the excel file
        REQUIERD_COL (list of strings): Name of required columns

    Returns:
        df (pandas.DataFrame): The loaded DataFrame
        basename (str) : The basename of the excel file
    """

    basename, ext = os.path.splitext(os.path.basename(table_path))
    df = pd.read_excel(table_path) if ext == ".xlsx" else pd.read_csv(table_path)
    
    # Check the format
    if not set(REQUIERD_COL).issubset(df.columns):
        print(f"A needed column is not found on the given file: {basename}")
        return KeyError

    # Del miss imported columns (named "Unnamed_N") 
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df, basename

def clean_productCode(df, MATRICES_TREE):
    """Delete rows with a non-conforme productCode according to the MATRICES tree given in the CONFIG json

    Args:
        df (pandas.DataFrame): The loaded DataFrame
        MATRICES_TREE (pandas.DataFrame): The matrices tree (all matrices to classify)

    Returns:
        df (pandas.DataFrame): The loaded DataFrame cleaned
    """
    
    # Delete rows with non-conform productCode
    valid_productCode = MATRICES_TREE["ProductCode"].to_list()
    df = df[df['ProductCode'].isin(valid_productCode)]

    return df

def _cleaning_func(sentence, lang_short, lemmatizer, stopwords, DEL_LIST, CONVERT_DICT):
         
        sentence = str(sentence).lower()
        sentence = unidecode(sentence)
        # Delete special character
        sentence = re.sub(r'[^a-z0-9àéâîïèœê%\s]', ' ', sentence)

        # Split the descriptions into single words and delete repeted words
        words = []
        [words.append(word) for word in str(sentence).split() if not word in words]
        
        res_words = []

        for word in words:

            # if number is a percent or a mass keep it else remove it
            if any(char.isdigit() for char in word):
                if word[-1] == "%":
                    word = re.sub(r',', '.', word)
                    number = re.sub(r'[^0-9\s|.]', '', word[:-1])
                    number = num2words(int(number), lang=lang_short)
                    word = number + " percent"
                
                elif word[-2:] == "mg":
                    word = re.sub(r',', '.', word)
                    number = re.sub(r'[^0-9\s|.]', '', word[:-2])
                    number = num2words(int(number), lang=lang_short)
                    word = number + " miligramme"
                
                elif word[-2:] == "kg":
                    word = re.sub(r',', '.', word)
                    number = re.sub(r'[^0-9\s|.]', '', word[:-2])
                    number = num2words(int(number), lang=lang_short)
                    word = number + " kilogramme"

                elif word[-1] == "g":
                    word = re.sub(r',', '.', word)
                    number = re.sub(r'[^0-9\s|.]', '', word[:-1])
                    number = num2words(int(number), lang=lang_short)
                    word = number + " gramme"
                else:
                    continue

            # Delete specified words
            if word in DEL_LIST or len(word)<=1 or word in stopwords:
                continue
            
            if word in  list(CONVERT_DICT.keys()):
                word = CONVERT_DICT[word]
            res_words.append(lemmatizer.lemmatize(word))

        # Generate a clean sentence
        cleaned_sentence = " ".join(res_words)
        
        return cleaned_sentence

def clean_description(df, LANG, DEL_LIST, CONVERT_DICT):
    """ Clean the description columns following rules (delete specific character, lemmatize, change usefull nulber to words...)

    Args:
        df (pandas.DataFrame): The loaded DataFrame
        LANG (dict of language name): with "short" for the abreviatoin and anf "full" for the full name as keys.
        DEL_LIST (list of string): Specific words to delete
        CONVERT_DICT (dict): Dictionnary with the word to change as key and the new word as item

    Returns:
        df (pandas.DataFrame): The loaded DataFrame with the cleanDescription columns
    """

    def _load_lemmatizer(LANG):
        """Load the lemmatazer and the stopwords list

            If you have SSL issues try : git config --global http.sslbackend schannel
        
        Args:
            LANG (dict of language name): with "short" for the abreviatoin and and "full" for the full name as keys. 

        Returns:
            spacy_model (spacy model): The spacy model according to the language
        """
        if LANG['short'] == "fr":
            from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
            lemmatizer = FrenchLefffLemmatizer()
            import nltk
            # nltk.download('stopwords')
            stopwords = nltk.corpus.stopwords.words(LANG["full"])
        
        return lemmatizer, stopwords
    
    
    # Load sentence lemma and stop words from NLP spacy model
    lemmatizer, stopwords = _load_lemmatizer(LANG)

    # Delete rows without description
    cleanDescription = df["Description"].apply(lambda x: _cleaning_func(x, LANG["short"], lemmatizer, stopwords, DEL_LIST, CONVERT_DICT))
    df["CleanDescription"] = deepcopy(cleanDescription)
    
    df = df[df['CleanDescription'] != '']

    return df

def merge_dataframes(df_list):
    """Merged dataframes

    Returns:
        df (pandas.DataFrame): The merged DataFrame
    """
    return pd.concat([df for df in df_list])

def arrange_cols(df, COL_ORDER):
    """Arrange columns according to the specified order

    Args:
        df (pd.DataFrame)
        COL_ORDER (list of columns as str)

    Returns:
        df (pd.DataFrame)
    """
    df_cols = df.columns

    # First set most important columns
    new_cols = [col for col in COL_ORDER if col in df_cols]

    # Then add other columns from the dataframe
    new_cols += [col for col in df_cols if not col in new_cols]

    df = df.reindex(new_cols, axis=1)

    return df

def clean_rows(merged_df, FEATURES_COL, TARGET_COL):
    
    # Del NA on the needed columns
    merged_df.dropna(axis=0, subset=FEATURES_COL)

    # Del similar SampleCode 
    merged_df.drop_duplicates(subset=FEATURES_COL)

    # Del similar SampleCode 
    merged_df.drop_duplicates(subset=TARGET_COL)

    return merged_df

def save_df(df, save_folder):
    """Save DataFrame at the save_path place.

    Args:
        df (pandas.DataFrame): a DataFrame
        save_folder (str folder path): the path
    """

    name = "Merged_and_cleand_df_" + str(date.today())
    for i in range(0, len(df),MAX_EXCEL_ROW):
        df.iloc[i:i+MAX_EXCEL_ROW-1,:].to_excel(os.path.join(save_folder, name+f"_{i}.xlsx"), index=False)

def process_multi_path(folder_path, config_path, save_folder=""):
    """Load, process and save excels into one unique DataFrame

    Args:
        args_path : Path .xlsx to all df to process

    """
    # Extract config
    DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

    # Load de matrices tree
    MATRICES_TREE, _ = load_df_from_excel(DATA_CONFIG["MATRICES_TREE_PATH"], DATA_CONFIG["REQUIERD_COL_TREE"])

    table_pathes = getAllTables(folder_path)
    print(f"Excel collected: {len(table_pathes)}")

    df_list = []

    # Cleaning pipeline
    for enum, table_path in enumerate(table_pathes):

        print(f"Process {enum}/{len(table_pathes)}")
        # If "Merged" is in the path it means that the current excel has already been process
        if "Merged" in table_path:
            print(f"{table_path} is not process")
            df, basename = load_df_from_excel(table_path, DATA_CONFIG["REQUIERD_COL_DF"])
            df_list.append(df)
        
        else:
            # Other tables are process
            df, basename = load_df_from_excel(table_path, DATA_CONFIG["REQUIERD_COL_DF"]) 
            df = clean_productCode(df, MATRICES_TREE)
            df = clean_description(df, DATA_CONFIG["LANG"], DATA_CONFIG["DEL_LIST"], DATA_CONFIG["CONVERT_DICT"])
            df["Laboratory"] = basename.split("_")[0]
            df_list.append(df)

    # Merged clean dfs
    merged_df = merge_dataframes(df_list)
    print("All df are merged")

    # Arrange columns and rows
    merged_df = arrange_cols(merged_df, DATA_CONFIG["COL_ORDER"])
    # Arrange columns and rows
    merged_df = clean_rows(merged_df, DATA_CONFIG["REQUIERD_COL_DF"], DATA_CONFIG["TARGET_COL"])
    print("All df are cleand")

    # Save the final df
    if save_folder:
        print(f"Saving the final df of size {len(merged_df)}")
        save_df(merged_df, save_folder)

    return merged_df

if __name__ == "__main__":

    print("-- START : samples extraction clean and merge --")

    version = "VERSION_NG_FR"

    config_path = r"Config\DataConfig.json"

    folder_path = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\to_process_and_merge"
    process_multi_path(folder_path, config_path, save_folder=r"C:\Users\CF6P\Desktop\PAC\Data_PAC")

    print("-- DONE --")

