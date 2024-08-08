import os
import shutil
import pandas as pd

def add_image_info_and_move(sample_df, image_folder):
    # Create new columns with default values
    sample_df['ImageName'] = ''
    sample_df['Extension'] = ''
    
    # Create the sample_scan folder if it doesn't exist
    sample_scan_folder = os.path.join(image_folder, 'sample_scan')
    os.makedirs(sample_scan_folder, exist_ok=True)

    listdir = os.listdir(image_folder)
    # Iterate through all files in the image folder
    for i, filename in enumerate(listdir):
        if i%5000==0:
            print(f"Iteration {i+1}/{len(listdir)}")

        file_path = os.path.join(image_folder, filename)
        file_path = os.path.join(image_folder, filename)
        
        # Check if it's a file and if it's an image (pdf or jpg)
        if os.path.isfile(file_path) and filename.lower().endswith(('.pdf', '.jpg')):
            # Extract the sample code from the filename
            sample_code = filename.split('_')[0]
            
            # Check if the sample code is in the DataFrame
            if sample_code in sample_df['SampleCode'].values:
                # Find the index of the corresponding row
                idx = sample_df.index[sample_df['SampleCode'] == sample_code].tolist()[0]
                
                # Add the image name and extension
                sample_df.at[idx, 'ImageName'] = filename
                sample_df.at[idx, 'Extension'] = os.path.splitext(filename)[1][1:]  # Remove the dot
                
                # Move the file to the sample_scan folder
                shutil.move(file_path, os.path.join(sample_scan_folder, filename))
    
    return sample_df

def add_name_into_df(sample_df, image_folder):
    # Create lists to store file information
    sample_codes = []
    image_names = []
    extensions = []
    
    listdir = os.listdir(image_folder)
    print(f"Total files to process: {len(listdir)}")
    
    # Iterate through all files in the image folder
    for i, filename in enumerate(listdir):
        if i % 5000 == 0:
            print(f"Iteration {i}/{len(listdir)}")
        
        # Check if it's an image file (pdf or jpg)
        if filename.lower().endswith(('.pdf', '.jpg')):
            # Extract the sample code from the filename
            sample_code = filename.split('_')[0]
            
            # Append the details to the lists
            sample_codes.append(sample_code)
            image_names.append(filename)
            extensions.append(os.path.splitext(filename)[1][1:])  # Remove the dot
    
    # Create a DataFrame from the extracted information
    image_info_df = pd.DataFrame({
        'SampleCode': sample_codes,
        'ImageName': image_names,
        'Extension': extensions
    })
    
    # Merge the sample_df with the image_info_df on the 'SampleCode' column
    result_df = pd.merge(sample_df, image_info_df, on='SampleCode', how='left')
    
    return result_df

if __name__ == "__main__":
    # Assuming you already have a DataFrame sample_df
    sample_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\Merged_and_cleand_df_2024-06-11_0.xlsx")
    image_folder = r'\\itiz-ft-fr.local\AAG010\Shared\DeepLearning\PACFiles'

    new_folder = r"\\itiz-ft-fr.local\AAG010\Shared\DeepLearning\PACFiles\sample_scan"

    updated_df = add_name_into_df(sample_df, new_folder)

    updated_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\IM_Merged_and_cleand_df_2024-06-11_0.xlsx")