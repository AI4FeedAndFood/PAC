#dépendance !pip install pdf2image !sudo apt-get install -y poppler-utils
# !pip install azure-ai-vision-imageanalysis
# PIL
#pandas
#!pip install paddleocr
#!pip install paddlepaddle
import os 
import pandas as pd 
from read_config import read_config_predict
### AZURE OCR ###
import pdf2image 
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from typing import Dict
from typing import Any 
import argparse

from constant import KEY_SECRET, ENDPOINT_SECRET
#TODO keep these informations secret 

def convert_pdf_as_image(pdf_path, output_folder = "/content/") -> list[bytes]:
    """
    This function converts a PDF file into images and deletes the temporarily created image files.

    Args:
        pdf_path (str): The path to the PDF file to be converted to images.
        output_folder (str, optional): The path to the folder where the temporary image files will be saved. Defaults to "/content/".

    Returns:
        images_data (list[bytes]): A list of image data in bytes, where each element represents an image from each pages of a PDF file.

    Note:
        The function uses the pdf2image library to convert the PDF to images.
        The temporarily created image files are deleted after they have been read into memory.
        If no images are generated from the PDF, a message is printed indicating that no images were generated.

    Example:
        images_data = convert_pdf_as_image("sample.pdf", "/content/output_folder")
        print(images_data)  # Prints the list of image data
    """
    
    if not os.path.isfile(pdf_path):
        print(f"PDF file '{pdf_path}' doesn't exist.")
        raise FileNotFoundError
    elif os.path.splitext(pdf_path)[1].lower() != '.pdf':
        raise ValueError("File must be a PDF.")
    else:
        images_path = pdf2image.convert_from_path(pdf_path, output_folder=output_folder, fmt='jpeg', paths_only=True)
        images_data = []
        for image_path in images_path:
            with open(image_path, "rb") as f:
                image_data= f.read()
                images_data.append(image_data)
                if os.path.isfile(image_path):
                    os.remove(image_path)
                    #print(f"Le fichier '{image_path}' a été supprimé avec succès.")
                else:
                    print(f"Le fichier '{image_path}' n'existe pas.")
        else:
            print("Aucune image n'a été générée à partir du PDF.")

        return images_data

def call_azure_ocr(image_data, log = False, name_image = "")-> Dict[str, Dict[str, Any]]:
    """
    This function calls Azure's OCR API to extract text from an image.

    Args:
        image_data (bytes): The image data in bytes.
        log (bool, optional): A flag indicating whether to print the OCR results to the console. Defaults to False.
        name_image (str, optional): The name of the image being analyzed. Defaults to "".

    Returns:
        ImageAnalysis (Dict[str, Dict[str: Any]]): The result of the OCR analysis, which includes the extracted text and metadata.

    Note:
        The function uses the Azure Image Analysis client to call the OCR API.
        The Azure endpoint and key are read from environment variables.
        If the environment variables are not set, an error message is printed and the function exits.

    Example:
        imageAnalysis = call_azure_ocr(image_data, log=True)
        print(imageAnalysis)  # Prints the OCR analysis result
    """
    try:
        endpoint = ENDPOINT_SECRET
        key = KEY_SECRET
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )


    # Get a caption for the image. This will be a synchronously (blocking) call.
    imageAnalysis = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    )

    print(f"OCR Call is complished for: {name_image}")
    # Print caption results to the console
    if log:
      show_result_OCRAzure(imageAnalysis)

    return imageAnalysis
  
def show_result_OCRAzure(imageAnalysis):
    """
    This function prints the OCR analysis result to the console.

    Args:
        imageAnalysis (ImageAnalysis): The result of the OCR analysis, which includes the extracted text and metadata.

    Returns:
        None

    Note:
        The function prints the caption, dense captions, and read text (OCR) analysis results to the console.

    Example:
        imageAnalysis = call_azure_ocr(image_data)
        show_result_OCRAzure(imageAnalysis)  # Prints the OCR analysis result to the console
    """
    if imageAnalysis.caption is not None:
        print(" Caption:")
        print(f"   '{imageAnalysis.caption.text}', Confidence {imageAnalysis.caption.confidence:.4f}")
    if imageAnalysis.dense_captions is not None:
        print(" Dense_captions:")
        print(f"   '{imageAnalysis.dense_captions}'")
    if imageAnalysis.read is not None:
        print(" Read:")
        for line in imageAnalysis.read.blocks[0].lines:
            print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            for word in line.words:
                print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")


def list_of_lines_from_OCRAzure(result, bouding_box = False) -> list[list[str]]:
    """
    This function extracts a list of lines of text from an Azure OCR result.

    Args:
        result (ImageAnalysis): The result of the OCR analysis, which includes the extracted text and metadata.
        bouding_box (bool, optional): TODO 

    Returns:
        list[list[str]]: A list of lists of strings, where each outer list represents a block of text and each inner list represents a line of text within the block.

    Note:
        The function extracts the text from the `read` attribute of the `result` object.

    Example:
        result = call_azure_ocr(image_data)
        lines = list_of_lines_from_OCRAzure(result)
        print(lines) 
        >>> [['La barquette de ce produit est',
          'Ingrédients' ]]
    """
    block_text= []
    for block in result.read.blocks:
        line_text = []
        for line in block.lines:
            line_text.append(line.text)
        block_text.append(line_text)
    return block_text

def list_of_words_from_OCRAzure(result, bouding_box = False, confidence = False) -> list[list[str]]:
    """
    This function extracts a list of words from an Azure OCR result.

    Args:
        result (ImageAnalysis): The result of the OCR analysis, which includes the extracted text and metadata.
        bouding_box (bool, optional): TODO
        confidence (bool, optional): TODO 

    Returns:
        list[list[str]]: A list of lists of strings, where each outer list represents a block of text and each inner list represents a word within the block.

    Note:
        The function extracts the text from the `read` attribute of the `result` object.
        If `bouding_box` is True, each word is represented as a tuple of the form (text, bounding box).
        If `confidence` is True, each word is represented as a tuple of the form (text, confidence score).

    Example:
        result = call_azure_ocr(image_data, log=True)
        words = list_of_words_from_OCRAzure(result, bouding_box=True, confidence=True)
        print(words)  # Prints the list of words and their bounding boxes and confidence scores
        >>> [[  'La',
        'barquette',
        'de',
        'ce',
        'produit',
        'est',
        'Ingrédients']]
    """
    block_text= []
    for block in result.read.blocks:
        word_text = []
        for line in block.lines:
            for word in line.words:
                word_text.append(word.text)
        block_text.append(word_text)
    return block_text

def list_str_to_str(list_str):
    """
    This function concatenates a list of strings into a single string.

    Args:
        list_str (list): A list of strings to be concatenated.

    Returns:
        str: A single string formed by concatenating the elements of `list_str` with a comma and a space separating each element.

    Example:
        list_str = ['Hello', 'world', '!']
        result = list_str_to_str(list_str)
        print(result)  # Prints 'Hello , world , !'
    """
    strings = ""
    for string in list_str:
        strings +=  str(string) + " , "
    return strings


def from_path_to_text_OCRAzure(path:str) -> str:
    """
    This function extracts text from a PDF file using Azure OCR API.

    Args:
        path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
        
    Raises:
        ValueError: If the file is not a PDF.

    Note:
        The function first converts the PDF file into a list of images using the `convert_pdf_as_image` function.
        Then, it calls the `call_azure_ocr` function for each image to extract text using Azure OCR API.
        Finally, it concatenates the extracted text from all images into a single string using the `list_str_to_str` function.

    Example:
        path = 'sample.pdf'
        text = from_path_to_text_OCRAzure(path)
        print(text)  # Prints the extracted text from the PDF file
    """
    
    images_data = convert_pdf_as_image(path)

    txt = ""
    for image_data in images_data:
        result = call_azure_ocr(image_data, name_image=path)
        block_text = list_of_lines_from_OCRAzure(result)
        for text in block_text:
            txt += list_str_to_str(text)

    return txt

def from_config_to_text(path_config):
    config = read_config_predict(path_config)
    file_data = config.get("file_data")
    text_ocr = from_path_to_text_OCRAzure(file_data)
    return text_ocr

def check_file_exists(file_path: str) -> bool:
    """
    Args:
        file_path (str): Le chemin d'accès au fichier à vérifier.

    Returns:
        bool: True si le fichier existe, False sinon.
    """
    return os.path.isfile(file_path)

def add_text_OCRAzure_to_dataset(dataset : pd.DataFrame):
    """
    This function adds a new column to a pandas DataFrame containing the extracted text from PDF files using Azure OCR API.
    The dataset need a column named 'path'

    Args:
        dataset (pd.DataFrame): The pandas DataFrame to which the new column will be added.

    Returns:
        None

    Raises:
        KeyError: If the 'path' column is not present in the DataFrame.
        FileExistsError: If some file(s) specified in the 'path' column do not exist.

    Note:
        The function assumes that the 'path' column contains the file paths of the PDF files.
        The function calls the `from_path_to_text_OCRAzure` function for each file path to extract text using Azure OCR API.
        The extracted text is added as a new column named 'Azure_text' to the DataFrame.

    Example:
        dataset = pd.DataFrame({'path': ['sample1.pdf', 'sample2.pdf', 'sample3.pdf']})
        add_text_OCRAzure_to_dataset(dataset)
        print(dataset)  # Prints the DataFrame with the new 'Azure_text' column
    """
    if not "path" in dataset.columns: 
        raise KeyError("'path' not a column in the dataset, cannot find the pdf/image paths to process OCR")
    else: 
        ocr_txts = []
        paths = dataset['path']
        raiseError = False
        for path in paths:
            if not check_file_exists(path):
                print(f"Warning {path} doesn't exist")
                raiseError = True 
        if raiseError:
            raise FileExistsError("Some file(s) don't exist.")
        for path in paths:
            ocr_txts.append(from_path_to_text_OCRAzure(path))
        
        dataset["Azure_text"] = ocr_txts


### PADDLE OCR ###
from paddleocr import PaddleOCR
import fitz
from PIL import Image
import numpy as np 

def PDF_to_images(path):
    """
    Open the pdf and return all pages as a list of array
    Args:
        path (path): python readable path

    Returns:
        list of arrays: all pages as array
    """
    images = fitz.open(path)
    res = []
    for image in images:
        pix = image.get_pixmap(dpi=300)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img = np.array(img)
        res.append(img)
    return res   

def from_path_to_text_PaddleOCR(path:str, ocr : PaddleOCR) -> str:
    """
    Extract text from a PDF file using PaddleOCR.

    Args:
        path (str): The path to the PDF file.
        ocr (PaddleOCR): The PaddleOCR object to use for OCR.

    Returns:
        str: The extracted text from the PDF file.

    Raises:
        ValueError: If the file is not a PDF.
    """
    if os.path.splitext(path)[1].lower() == '.pdf':
        images = PDF_to_images(path)
        images = images[0:]
        txts = ""
        for im in images:
            results = ocr.ocr(im, cls=True) 
            for result in results :
                if result != None:
                    for line in result:
                            txts +=  " " + line[1][0]
                else:
                    print(f"Warning: PaddleOCR from {path} results to a partial/global None result, result: {txts}")
        return txts 
    else:
        raise ValueError("File must be a PDF.")

def add_text_PaddleOCR_to_dataset(dataset : pd.DataFrame):
    """
    Add a new column to a Pandas DataFrame containing the text extracted from PDF files using PaddleOCR.

    Args:
        dataset (pd.DataFrame): The Pandas DataFrame to which the new column will be added.

    Raises:
        KeyError: If the 'path' column is not present in the DataFrame.
        FileExistsError: If some file(s) specified in the 'path' column do not exist.
    """
    if not "path" in dataset.columns: 
        raise KeyError("'path' not a column in the dataset, cannot find the pdf/image paths to process OCR")
    else: 
        ocr = PaddleOCR(use_angle_cls=True, lang='french')
        ocr_txts = []
        paths = dataset['path']
        raiseError = False
        for path in paths:
            if not check_file_exists(path):
                print(f"Warning {path} doesn't exist")
                raiseError = True 
        if raiseError:
            raise FileExistsError("Some file(s) don't exist.")
        for path in paths:
            ocr_txts.append(from_path_to_text_PaddleOCR(path, ocr))
        
        dataset["Paddle_text"] = ocr_txts


if __name__ == "__main__":
    # df = pd.read_csv("/content/drive/MyDrive/Data/set1-pdf/data1-1.csv",index_col=False)
    # add_text_OCRAzure_to_dataset(df)
    # df.to_csv("/content/drive/MyDrive/Data/set1-pdf/data1-1.csv",index=False)
    parser = argparse.ArgumentParser(description="OCR call")

    parser.add_argument("path_pdf", type=str, help="Le chemin d'accès au fichier PDF d'entrée")

    args = parser.parse_args()

    text = from_path_to_text_OCRAzure(args.path_pdf)

    print(text)