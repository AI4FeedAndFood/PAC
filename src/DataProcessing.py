import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from DataProcess.RawDataPreprocess import load_data, process_and_encode_dataframe, set_config
from DataProcess.ImageMultiProcess import process_excel_files
from DataProcess.ReleventFilter import select_relevent_descriptions
from DataProcess.SplitData import add_augmented_data, split_data

VERSION = "VERSION_NG_FR"
CONFIG_PATH = r"Config\DataConfig.json"
DATA_CONFIG = set_config(CONFIG_PATH, VERSION)

def save_df(df, output_folder, filename):
    """Save DataFrame to a specified output folder."""
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"processed_{filename}")

    if len(df) > 1_048_576:
        chunk_size = 1_000_000
        for i, chunk in enumerate(pd.read_csv(df.to_csv(index=False), chunksize=chunk_size)):
            chunk_path = os.path.join(output_folder, f"processed_{filename}_part{i+1}.xlsx")
            chunk.to_excel(chunk_path, index=False, engine='openpyxl')
            print(f"DataFrame part {i+1} saved to {chunk_path}")
    else:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"DataFrame saved to {output_path}")

def load_output_folder_df(file_path, output_folder):
    """Load DataFrame from a specified output folder."""

    filename = os.path.basename(file_path)
    base_filename, _ = os.path.splitext(filename)
    search_pattern = f"processed_{base_filename}"

    files_in_out_folder = [f for f in os.listdir(output_folder) if search_pattern in f]
    
    if not files_in_out_folder:
        return pd.DataFrame()

    dfs = [pd.read_excel(os.path.join(output_folder, file), engine='openpyxl') for file in sorted(files_in_out_folder)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_file(file_path, data_config):
    """Process a single file and return a DataFrame."""
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    laboratory = os.path.basename(file_path).split("_")[0]
    df = process_and_encode_dataframe(df, data_config, laboratory=laboratory)
    df["OriginDF"] = os.path.basename(file_path)
    return df

def process_data(input_path, output_folder, data_config):
    """Process input data (single file or folder) and apply necessary transformations."""

    if os.path.isfile(input_path):
        merged_df = load_output_folder_df(input_path, output_folder)
        if merged_df.empty:
            merged_df = process_file(input_path, data_config)
            save_df(merged_df, output_folder, os.path.basename(input_path))
    
    elif os.path.isdir(input_path):
        all_files = load_data(input_path)
        all_dfs = []
        for file in tqdm(all_files, desc="Processing files"):
            df = load_output_folder_df(file, output_folder)
            if df.empty:
                print(f"\n_____ {os.path.basename(file)} _____")
                df = process_file(file, data_config)
                save_df(df, output_folder, os.path.basename(file))
            all_dfs.append(df)
        merged_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raise ValueError("Input must be a file or a directory")

    print("Merged DataFrame is set")
    return merged_df

def apply_mode_processing(df, mode, data_config,p_valid=0.1):
    """Apply mode-specific processing to the DataFrame."""

    if mode in ["valid", "test", "train"]:
        df["Mode"] = mode

    elif mode in ["train_test", "train_test_valid"]:
        print(f"Train test split is selected - merged length: {len(df)}")
        valid_df = df[df["Mode"] == "valid"] if "Mode" in df.columns else None
        df = df[df["Mode"] != "valid"] if valid_df is not None else df

        if mode == "train_test_valid":
            # Calculer l'index des 10% dernières lignes pour chaque groupe
            valid_indices = df.groupby("OriginDF").apply(
                lambda x: x.index[-int(len(x) * p_valid):]
            ).explode()

            # Créer valid_df et marquer ces lignes
            valid_df = df.loc[valid_indices].copy()
            valid_df["Mode"] = "valid"

            # Supprimer ces lignes du DataFrame principal
            df = df.drop(valid_indices)

        df = add_augmented_data(df, data_config, aug_thresh=10)
        df = select_relevent_descriptions(df)
        print(f"Relevant descriptions are selected - merged length: {len(df)}")

        df = split_data(df, test_size=0.2, seed=42)
        print(f"Split is done - merged length: {len(df)}")
        
        if valid_df is not None:
            df = pd.concat([df, valid_df], ignore_index=True)
            print(f"All merged data with valid is set - merged length: {len(df)}")

    return df

def main_text(input_path, mode, output_folder=None):
    """Main function to process the data."""

    if mode not in ["valid", "test", "train", "train_test", "train_test_valid"]:
        raise ValueError("Invalid mode. Choose from 'valid', 'test', 'train', 'train_test', or 'train_test_valid'")

    output_folder = output_folder or (os.path.dirname(input_path) if os.path.isfile(input_path) else input_path)

    print("1 - Processing all data")
    merged_df = process_data(input_path, output_folder, DATA_CONFIG)
    print(f"2 - Process {mode}")
    merged_df = apply_mode_processing(merged_df, mode, DATA_CONFIG)

    print("3 - Save")
    if os.path.isdir(input_path):
        save_df(merged_df, output_folder, "merged_processed_data.xlsx")

    return merged_df

def main_image(input_path, image_folder):
    process_excel_files(input_path, image_folder, max_workers=None)

if __name__ == "__main__":
    input_path = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\0_raw_extraction"
    mode = "train_test_valid"
    output_folder = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\0_no_image_data"

    try:
        result_df = main_text(input_path, mode, output_folder)
        print(f"Processing completed. DataFrame shape: {result_df.shape}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")