import pandas as pd
import shutil
import os 
import glob
from PyPDF2 import PdfMerger
import argparse
from datetime import datetime

#!python /content/process_datas_for_dataset_creation.py "/content/drive/MyDrive/Data/data_test/test_original" "/content/output"

def get_filenames(directory):
    """
    Retrieves all filenames in the specified directory.

    Parameters:
    - directory (str): The directory from which to retrieve filenames.

    Returns:
    - list: A list of full file paths for files in the directory.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def get_file_format(filename):
    """
    Gets the file format (extension) of a given filename.

    Parameters:
    - filename (str): The name of the file.

    Returns:
    - str: The file extension in lowercase.
    """
    return os.path.splitext(filename)[1].lower()

def is_file_smaller_than_5kb(filename):
    """
    Checks if a file is smaller than 5KB.

    Parameters:
    - filename (str): The name of the file.

    Returns:
    - bool: True if the file is smaller than 5KB, otherwise False.
    """
    file_size = os.path.getsize(filename)
    smaller = file_size < 5 * 1024
    if smaller:
        print(f"Warning, {filename} is smaller than 5kb. It will not be added to the dataset.")
    return smaller

def is_a_scan(filename):
    """
    Checks if a file is a scan based on its naming convention.

    Parameters:
    - filename (str): The name of the file.

    Returns:
    - bool: True if the file is identified as a scan, otherwise False.
    """
    filename = os.path.basename(filename)
    return filename.split("_")[1][:3].lower() == "img"

def process_pdfs(input_dir, output_dir, date_extraction, extraction_window_start, extraction_window_end, lab, env):
    """
    Processes PDF files from the input directory and copies them to the output directory.
    Creates a DataFrame with metadata about the processed files.

    Parameters:
    - input_dir (str): The directory containing the input PDF files.
    - output_dir (str): The directory to which processed PDF files will be copied.
    - date_extraction (str): The date of extraction.
    - extraction_window_start (str): The start date of the extraction window.
    - extraction_window_end (str): The end date of the extraction window.
    - lab (str): The lab identifier.
    - env (str): The environment identifier.

    Returns:
    - DataFrame: A DataFrame containing metadata about the processed PDF files.
    """    
    filenames = get_filenames(input_dir)
    filenames_pdf = [f for f in filenames if get_file_format(f) == '.pdf' and not is_file_smaller_than_5kb(f) and is_a_scan(f)]
    df_pdf = pd.DataFrame()
    paths, dates, sampleCodes, names = [], [], [], []

    for filename in filenames_pdf:
        new_file_path = os.path.join(output_dir, os.path.basename(filename))
        shutil.copy2(filename, new_file_path)
        
        name_file = os.path.basename(new_file_path)
        names.append(name_file)
        logs = name_file.split("_")
        sampleCode = logs[0]
        
        if logs[1] == "IMG":
            date = f"{logs[2][6:8]}/{logs[2][4:6]}/{logs[2][:4]}"
        else:
            date_ = logs[1][3:]
            date = f"{date_[6:8]}/{date_[4:6]}/{date_[:4]}"
        
        paths.append(new_file_path)
        dates.append(date)
        sampleCodes.append(sampleCode)

    df_pdf["path"] = paths
    df_pdf["name"] = names
    df_pdf["dates"] = dates
    df_pdf["sampleCodes"] = sampleCodes
    df_pdf['date_extraction'] = date_extraction
    df_pdf['extraction_window_start'] = extraction_window_start
    df_pdf['extraction_window_end'] = extraction_window_end
    df_pdf['lab'] = lab
    df_pdf['env'] = env

    return df_pdf

def concatenate_pdfs(df, output_dir):
    """
    Concatenates PDF files belonging to the same sample code and saves the concatenated PDF to the output directory.

    Parameters:
    - df (DataFrame): The DataFrame containing metadata about the PDF files.
    - output_dir (str): The directory to save the concatenated PDF files.

    Returns:
    - DataFrame: The updated DataFrame with paths to the concatenated PDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    updated_paths = []

    for samplecode, group in df.groupby('sampleCodes'):
        files = group['path'].tolist()
        
        if len(files) > 1:
            merger = PdfMerger()
            for pdf in files:
                merger.append(pdf)

            concatenated_pdf_path = os.path.join(output_dir, f"{samplecode}_concatenated.pdf")
            merger.write(concatenated_pdf_path)
            merger.close()

            updated_paths.extend([concatenated_pdf_path] * len(files))

            # Remove original files after concatenation
            for pdf in files:
                os.remove(pdf)
        else:
            updated_paths.extend(files)

    df['path'] = updated_paths
    return df

def main():
    parser = argparse.ArgumentParser(description="Process and concatenate PDF files.")
    parser.add_argument("input_pdf", help="Input directory containing PDF files")
    parser.add_argument("output_pdf", help="Output directory for processed PDF files")
    parser.add_argument("--date_extraction", default=datetime.now().strftime("%d/%m/%Y"), help="Date of extraction")
    parser.add_argument("--extraction_window_start", default=(datetime.now() - pd.Timedelta(days=4)).strftime("%d/%m/%Y"), help="Start of extraction window")
    parser.add_argument("--extraction_window_end", default=datetime.now().strftime("%d/%m/%Y"), help="End of extraction window")
    parser.add_argument("--lab", default="EUFR110", help="Lab identifier")
    parser.add_argument("--env", default="PRD42", help="Environment identifier")

    args = parser.parse_args()

    # Process PDFs
    df_pdf = process_pdfs(args.input_pdf, args.output_pdf, args.date_extraction, args.extraction_window_start, args.extraction_window_end, args.lab, args.env)

    # Concatenate PDFs if necessary
    df_pdf = concatenate_pdfs(df_pdf,  args.output_pdf)

    # Save the final DataFrame to a CSV file
    csv_output = os.path.join(args.output_pdf, "processed_pdfs.csv")
    df_pdf.to_csv(csv_output, index=False)

    print(f"PDF processing and concatenation completed. Results saved in {csv_output}")

if __name__ == "__main__":
    main()