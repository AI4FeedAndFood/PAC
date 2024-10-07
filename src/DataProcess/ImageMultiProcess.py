import os
import pandas as pd
import fitz
import time

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_excel(excel_file, image_folder):
    """
    Process an Excel file, associating images with sample codes and handling various image scenarios.

    Args:
    excel_file (str): Path to the Excel file to process.
    image_folder (str): Path to the folder containing image files.

    Returns:
    str: A message indicating success or failure of the processing.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Check if the SampleCode column exists
        if 'SampleCode' not in df.columns:
            print(f"Error: 'SampleCode' column missing in {excel_file}")
            return None

        # Create a dictionary to store images for each sample code
        image_dict = {}
        for image_file in os.listdir(image_folder):
            sample_code, ext = os.path.splitext(image_file)
            sample_code = sample_code.split("_")[0]
            
            # Only process image files and PDFs
            if ext.lower() in [".jpg", ".jpeg", ".png", ".pdf"]:
                if sample_code not in image_dict:
                    image_dict[sample_code] = []
                image_dict[sample_code].append(image_file)

        def process_images(sample_code):
            """
            Process images for a given sample code, handling different scenarios.

            Args:
            sample_code (str): The sample code to process images for.

            Returns:
            list: A list of image names (up to 10) associated with the sample code.
            """
            if sample_code not in image_dict:
                return [''] * 10  # Return 10 empty columns if no images

            images = image_dict[sample_code]
            result = [''] * 10  # Initialize 10 columns
            image_index = 0

            for image in images:
                _, ext = os.path.splitext(image)
                if ext.lower() == '.pdf':
                    # Convert PDF to images using PyMuPDF
                    pdf_path = os.path.join(image_folder, image)
                    pdf_document = fitz.open(pdf_path)
                    for page_num, page in enumerate(pdf_document):
                        pix = page.get_pixmap()
                        jpg_filename = f"{sample_code}_page_{page_num+1}.jpg"
                        jpg_path = os.path.join(image_folder, jpg_filename)
                        pix.save(jpg_path)
                        if image_index < 10:
                            result[image_index] = jpg_filename
                            image_index += 1
                    pdf_document.close()
                    # Remove the original PDF
                    os.remove(pdf_path)
                else:
                    # Process regular image files
                    if image_index < 10:
                        result[image_index] = image
                        image_index += 1

            return result

        # Create ImageName columns
        for i in range(1, 11):
            df[f'ImageName{i}'] = ''

        # Process images for each sample code
        df[['ImageName1', 'ImageName2', 'ImageName3', 'ImageName4', 'ImageName5',
            'ImageName6', 'ImageName7', 'ImageName8', 'ImageName9', 'ImageName10']] = df['SampleCode'].apply(process_images).tolist()

        # Save the updated Excel file
        df.to_excel(excel_file, index=False)
        return f"Processed: {excel_file}"
    except Exception as e:
        return f"Error processing {excel_file}: {str(e)}"

def process_excel_files(input_path, image_folder, max_workers=None):
    """
    Process multiple Excel files in parallel, updating each with the corresponding image information.

    Args:
    excel_folder (str): Path to the folder containing Excel files.
    image_folder (str): Path to the folder containing image files.
    max_workers (int, optional): Maximum number of worker processes to use. Defaults to None.

    Returns:
    None
    """
    start_time = time.time()
    if os.path.isfile(input_path) and input_path.lower().endswith(('.xlsx', '.xls')):
        excel_files = [input_path]
  
    elif os.path.isdir(input_path):
        excel_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(('.xlsx', '.xls'))]
    else:
        raise ValueError("Input must be a file or a directory")

    total_files = len(excel_files)

    print(f"Total number of Excel files to process: {total_files}", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_excel, excel_file, image_folder) 
                   for excel_file in excel_files]
        
        with tqdm(total=total_files, desc="Progress") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    print(result, flush=True)
                pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_file = total_time / total_files if total_files > 0 else 0

    print(f"\nProcess completed in {total_time:.2f} seconds", flush=True)
    print(f"Average time per file: {avg_time_per_file:.4f} seconds", flush=True)
    print(f"Total number of files processed: {total_files}", flush=True)

if __name__ == "__main__":
    excel_folder = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\0_raw_extraction"
    image_folder = r"C:\Users\CF6P\Desktop\PAC\Data_PAC\scans"
    process_excel_files(excel_folder, image_folder)