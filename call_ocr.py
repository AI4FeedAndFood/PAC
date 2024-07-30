# Dependencies and imports
import os 
import pandas as pd 
from read_config import read_config_predict
import argparse
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any 
import io 
import multiprocessing
from constant import KEY_SECRET, ENDPOINT_SECRET, MIN_LEN_TEXT_OCR_FOR_CROP
from crop_with_yolo import crop_with_yolo
from PIL import Image
import concurrent.futures
from functools import partial
from load_img_pdf import load_image, is_image_too_large, load_image_resized, convert_with_pdf2image

def pil_to_array(pil_image):
    """Convert a PIL image object to a byte array.

    Args:
        pil_image (PIL.Image): Pillow image object.

    Returns:
        bytes: PIL image object in the form of a byte array.
    """
    image_byte_array = io.BytesIO()
    pil_image.save(image_byte_array, format='PNG')
    image_data = image_byte_array.getvalue()
    return image_data


    
def call_azure_ocr(image_data, log=False, name_image="") -> Dict[str, Dict[str, Any]]:
    """
    Call Azure's OCR API to extract text from an image.

    Args:
        image_data (Image.Image): The image data as a PIL Image object.
        log (bool, optional): A flag indicating whether to print the OCR results to the console. Defaults to False.
        name_image (str, optional): The name of the image being analyzed. Defaults to "".

    Returns:
        Dict[str, Dict[str, Any]]: The result of the OCR analysis, which includes the extracted text and metadata.

    Raises:
        KeyError: If the required environment variables are not set.

    Example:
        image = Image.open("sample.jpg")
        result = call_azure_ocr(image, log=True, name_image="sample_image")
        print(result)  # Prints the OCR analysis result
    """
    try:
        endpoint = ENDPOINT_SECRET
        key = KEY_SECRET
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    image_bytes = pil_to_array(image_data) 
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Perform OCR analysis
    imageAnalysis = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    )

    print(f"OCR Call is accomplished for: {name_image}")
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
        list_str = [['Hello', 'world', '!']]
        result = list_str_to_str(list_str)
        print(result)  # Prints 'Hello , world , !'
    """
    strings = ""
    for list_ in list_str:
        for string in list_:
            strings +=  str(string) + " , "
        return strings





def from_path_to_text_OCRAzure(path: str, model_yolo_path=None, bool_is_crop=False) -> str:
    """
    Extract text from a PDF or image file using Azure OCR API.

    Args:
        path (str): The path to the PDF or image file.
        model_yolo_path (str, optional): Path to the YOLO model for image cropping. Defaults to None.
        bool_is_crop (bool, optional): Whether to return crop information. Defaults to False.

    Returns:
        str: The extracted text from the file.
        tuple: A tuple containing the extracted text and a boolean indicating if the image was cropped (if bool_is_crop is True).

    Raises:
        ValueError: If the file is neither a PDF nor a supported image format.

    Example:
        text = from_path_to_text_OCRAzure("sample.pdf")
        print(text)  # Prints the extracted text from the PDF file
    """
    # Load and preprocess the image(s)
    if model_yolo_path is not None:
        #If a yolo model is parsed, use it to crop the image
        images_data_crop, images_data_not_crop = crop_with_yolo(path, model_yolo_path)
        if len(images_data_crop) > 0:
            #if it detected a table 
            is_crop = True
            images_data = images_data_crop
        else:
            images_data = images_data_not_crop
            is_crop = False
    else:
        images_data = load_image(path)
        is_crop = False
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(process_single_image, image_data, path): image_data for image_data in images_data}
        results = []
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')


    text_ocr = ""
    for result in results: #result can be a string or None (don't use .join() method)
        if result != None:
            text_ocr += result 
    # Handle cases where YOLO cropping doesn't produce enough text
    if is_crop and len(text_ocr) < MIN_LEN_TEXT_OCR_FOR_CROP:
        print(f"Warning: YOLO detected an image and cropped it, but OCR didn't extract enough characters. Retrying with the full scan.")
        images_data = images_data_not_crop 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_image = {executor.submit(process_single_image, image_data, path): image_data for image_data in images_data}
            results = []
            for future in concurrent.futures.as_completed(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
        is_crop = False 
        text_ocr = ""
        for result in results:
            if result != None:
                text_ocr += result 
               
    if bool_is_crop:
        return text_ocr, is_crop
    else: 
        return text_ocr


def from_config_to_text(path_config):
    """
    Extract text from a PDF or image file using, all information have to be in the config file.
    Config file must be in the format: 
    {
        "model_yolo": "/content/best_mono_class.pt",
        "file_data" : "/content/968-2024-00046116_img20240330_11251323.pdf",
    }
    Args:
        path_config (str): The path to the config file.

    Returns:
        str: The extracted text from the file.
    """    
    config = read_config_predict(path_config)
    file_data = config.get("file_data")
    model_yolo_path = config.get("model_yolo")
    return from_path_to_text_OCRAzure(file_data, model_yolo_path)

#### THREADS 
import concurrent.futures
from functools import partial

def process_single_image(image_data, path=None):
    """
    Process a single image with Azure OCR.

    Args:
        image_data (bytes): The image data as bytes.
        path (str, optional): The path of the image file. Defaults to None.

    Returns:
        str: The extracted text from the image.

    Raises:
        ValueError: If the image is too large and cannot be resized.

    Example:
        with open("image.jpg", "rb") as f:
            image_data = f.read()
        text = process_single_image(image_data, "image.jpg")
        print(text)  # Prints the extracted text from the image
    """
    if is_image_too_large(image_data):
        print(f"Image is too large")
        resized_image = load_image_resized(image_data)
        if is_image_too_large(resized_image):
            raise ValueError("Unable to resize image to acceptable size")
        image_data = resized_image

    result = call_azure_ocr(image_data)
    block_text = list_of_lines_from_OCRAzure(result)
    return list_str_to_str(block_text)

def from_images_to_text(images_data, max_workers=5):
    """
    Convert a list of images to text using Azure OCR in parallel.

    Args:
        images_data (list[Image.Image]): A list of PIL Image objects to be processed.
        max_workers (int, optional): The maximum number of worker threads to use. Defaults to 5.

    Returns:
        str: The concatenated text extracted from all images.

    Raises:
        Exception: If an error occurs during the OCR process for an image.

    Example:
        images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
        text = from_images_to_text(images)
        print(text)  # Prints the extracted text from all images
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_image = partial(process_single_image)
        future_to_image = {executor.submit(process_image, image_data): image_data for image_data in images_data}
        
        txt = ""
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                txt += future.result()
            except Exception as exc:
                print(f'An image generated an exception: {exc}')

    return txt

def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.

    Example:
        if check_file_exists("sample.pdf"):
            print("The file exists")
        else:
            print("The file does not exist")
    """
    return os.path.isfile(file_path)

def add_text_OCRAzure_to_dataset(dataset : pd.DataFrame):
    """
    This function adds a new column to a pandas DataFrame containing the extracted text from PDF files using Azure OCR API.
    The dataset need a column named 'path'

    Args:
        dataset (pd.DataFrame): The pandas DataFrame to which the new column will be added.

    Returns:
        This function adds a new column named 'Azure_text' to the DataFrame.

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
    n = len(dataset)
    if not "path" in dataset.columns: 
        raise KeyError("'path' not a column in the dataset, cannot find the pdf/image paths to process OCR")
    else: 
        ocr_txts = []
        paths = dataset['path']
        raiseError = False
        
        k = 0 
        
        for path in paths:
            
            if not check_file_exists(path):
                print(f"Warning {path} doesn't exist")
                raiseError = True 
        if raiseError:
            raise FileExistsError("Some file(s) don't exist.")
        for path in paths:
            ocr_txts.append(from_path_to_text_OCRAzure(path))
            k +=1 
            print(f"{k}/{n}")
        
        dataset["Azure_text"] = ocr_txts
 
 
 #### Testing       
import unittest
from azure.ai.vision.imageanalysis.models import ImageAnalysisResult

class TestOCRFunctions(unittest.TestCase):
    def test_pil_to_array(self):
        image = Image.new('RGB', (100, 100), color='red')
        array = pil_to_array(image)
        self.assertIsInstance(array, bytes)
        
    def test_call_azure_ocr(self):
        # This test requires Azure credentials to be set up
        image = Image.new('RGB', (100, 100), color='white')
        result = call_azure_ocr(image, log=False, name_image="test_image")
        self.assertIsInstance(result, ImageAnalysisResult)


if __name__ == "__main__":
    unittest.main()