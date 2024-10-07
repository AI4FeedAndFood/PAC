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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ModelTools.Models import CustomBertEncoder

from transformers import AutoTokenizer

import json
import pandas as pd
import re
import torch

from tqdm import tqdm

from copy import deepcopy
from unidecode import unidecode

from num2words import num2words

MAX_EXCEL_ROW = 1048574

bert_model_name = 'sentence-transformers/all-distilroberta-v1'
TOKENIZER = AutoTokenizer.from_pretrained(bert_model_name, )
TEXT_ENCODER = CustomBertEncoder(bert_model_name, n_attention_layers=4)

def load_data(path, ext_list=[".csv", ".xlsx"]):
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
        print(f"A needed column is not found on the given file: {basename}\n- {df.columns}")
        return KeyError

    # Del miss imported columns (named "Unnamed_N") 
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df, basename

def clean_target_column(df, matrix_tree_df, TARGER_COL):
    """Delete rows with a non-conforme productCode according to the MATRICES tree given in the CONFIG json

    Args:
        df (pandas.DataFrame): The loaded DataFrame
        MATRICES_TREE (pandas.DataFrame): The matrices tree (all matrices to classify)

    Returns:
        df (pandas.DataFrame): The loaded DataFrame cleaned
    """
    
    df.dropna(axis=0, subset=TARGER_COL)
    # Apply the mapping to convert the product codes
    code_mapping = {code.split(".")[-1]: code for code in matrix_tree_df[TARGER_COL[0]] if code}
    product_code_unit = list(code_mapping.keys())
    df[TARGER_COL[0]] = df[TARGER_COL[0]].apply(lambda x: code_mapping.get(x, x) if x in product_code_unit else "unknown")
    # Delete rows with non-conform productCode
    df = df[df[TARGER_COL[0]].str.contains(".")]
    
    return df

def clean_rows(merged_df, UNICITY_COL, FEATURES_COL):
    
    # Del NA on the needed columns
    merged_df.dropna(axis=0, subset=FEATURES_COL+UNICITY_COL)

    # Del similar SampleCode 
    merged_df.drop_duplicates(subset=UNICITY_COL)

    # Del similar Features tuples 
    merged_df.drop_duplicates(subset=FEATURES_COL)

    return merged_df

def _cleaning_func(sentence, lang_short, lemmatizer, stopwords, DEL_LIST, CONVERT_DICT):
         
        sentence = str(sentence).lower()
        sentence = unidecode(sentence)
        # Delete special character
        sentence = re.sub(r'[^a-z0-9àéâîïèœê%\s]', ' ', sentence)
        # Split the descriptions into single words and delete repeted words
        words = []
        words = [word for word in str(sentence).split() if not word in words]
        
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

def clean_description(df, LANG, DEL_LIST=[], CONVERT_DICT={}):
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
    df = df.assign(CleanDescription=df["Description"].apply(
        lambda x: _cleaning_func(x, LANG["short"], lemmatizer, stopwords, DEL_LIST, CONVERT_DICT)
    ))
    df = df[df['CleanDescription'] != '']

    return df

def encode_description(df, tokenizer, text_encoder, device, batch_size=32):
    """
    Tokenize and encode the 'CleanDescription' column of a DataFrame using BERT.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'CleanDescription' column.
        tokenizer: BERT tokenizer.
        text_encoder: BERT encoder model.
        device (str): Device to use for encoding ('cuda' or 'cpu').
        batch_size (int): Number of samples to process in each batch.

    Returns:
        pd.DataFrame: DataFrame with added 'TokenizedCleanDescription' and 'EncodedCleanDescription' columns.
    """
    all_tokenized = []
    all_encoded = []
    
    # Convert the column to a list for batch processing
    texts = df['CleanDescription'].tolist()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the batch
        tokenized_batch = tokenizer(batch_texts,
                                    add_special_tokens=True,
                                    max_length=256,
                                    return_token_type_ids=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
        
        # Store tokenized results
        all_tokenized.extend([dict(zip(tokenized_batch.keys(), [t[i] for t in tokenized_batch.values()]))
                              for i in range(len(batch_texts))])
        
        # Encode the batch if CUDA is available
        if torch.cuda.is_available() and "cuda" in device:
            tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
            with torch.no_grad():
                encoded_batch = text_encoder(**tokenized_batch)
            encoded_batch = encoded_batch.last_hidden_state[:, 0, :].cpu().numpy()
            all_encoded.extend(encoded_batch)
        
    # Add columns to the DataFrame
    df["TokenizedCleanDescription"] = all_tokenized
    if torch.cuda.is_available() and "cuda" in device:
        df["EncodedCleanDescription"] = all_encoded
    else:
        df["EncodedCleanDescription"] = None
    
    return df

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

def process_and_encode_dataframe(df, DATA_CONFIG, laboratory="", save_folder=""):
    """
    Prépare un DataFrame pour l'utilisation dans un modèle en appliquant plusieurs étapes de nettoyage et de transformation.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant les données brutes.
        laboratory (str): Nom du laboratoire traitant les données.
        config_path (str): Chemin vers le fichier de configuration JSON.
        save_folder (str, optional): Dossier où sauvegarder le DataFrame traité. Par défaut "".

    Returns:
        pd.DataFrame: DataFrame nettoyé et préparé pour le modèle.
    """
    # Extraction de la configuration

    # Nettoyage des lignes inutilisables
    MATRIX_TREE_DF, _ = load_df_from_excel(DATA_CONFIG["MATRICES_TREE_PATH"], DATA_CONFIG["REQUIERD_COL_TREE"])
    df = clean_target_column(df, MATRIX_TREE_DF, DATA_CONFIG["TARGET_COL"])
    df = clean_rows(df, DATA_CONFIG["REQUIERD_COL_DF"], DATA_CONFIG["UNICITY_COL"])

    # Traitement de la description
    df = clean_description(df, DATA_CONFIG["LANG"], DATA_CONFIG["DEL_LIST"], DATA_CONFIG["CONVERT_DICT"])
    
    # Ajout de colonnes importantes : Laboratory, TokenizedCleanDescription et EncodedCleanDescription
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if laboratory:
        df["Laboratory"] = laboratory

    df = encode_description(df, TOKENIZER, TEXT_ENCODER, device, batch_size=32)

    # Réorganisation des colonnes
    df = arrange_cols(df, DATA_CONFIG["COL_ORDER"])

    return df

if __name__ == "__main__":

    df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\0_raw_extraction\EUBSCO_NG_Samples_2023.xlsx")
    DATA_CONFIG = {

        "LANG" : {
            "short" : "fr",
            "full" : "french"
        },
        
        "REQUIERD_COL_DF" :  ["AccountCode", "Description", "ProductCode", "SampleCode"],

        "REQUIERD_COL_TREE" :  ["ProductCode", "Nom"],

        "FEATURES_COL"  : ["AccountCode", "CleanDescription", "Laboratory"],

        "TARGET_COL" : ["ProductCode"],

        "COL_ORDER" : ["SampleCode", "AccountCode", "Description", "CleanDescription", "ProductName", "ProductCode"],

        "UNICITY_COL" : ["SampleCode"],

        "MATRICES_TREE_PATH" : "Config\\matrices_tree.xlsx",
        
        "DEL_LIST" : [],

        "CONVERT_DICT" : {}
    }

    df =  process_and_encode_dataframe(df, DATA_CONFIG, laboratory="")
    