# client.py
"""
This script implements a client for Estimates prediction API.
It handles OCR text processing, API communication, and result processing.
"""

import requests
import argparse
from call_ocr import from_config_to_text
from constant import COLUMNS
import pandas as pd 
from controle import check_from_config, create_result_table
import time 

def predict_estimates_from_api(text_ocr, api_url="http://127.0.0.1:8000/predict_estimates/"):
    """
    Send a prediction request to the API.
    
    Args:
        text_ocr (str): The OCR text to be processed.
        api_url (str): The URL of the prediction API.
    
    Returns:
        list or None: The prediction result if successful, None otherwise.
    """
    data = {"text_ocr": text_ocr}
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
def client_pipeline(path_config):
    """
    Execute the full client pipeline: OCR, prediction, and result processing.
    
    Args:
        path_config (str): Path to the configuration file.
    """
    starting_time = time.time()
    
    # Perform OCR
    text_ocr = from_config_to_text(path_config)
    
    ocr_time = time.time()    
    print(f"Time for calling OCR: {ocr_time - starting_time:.2f}s")
    print(text_ocr)
    
    # Get prediction from API
    prediction = predict_estimates_from_api(text_ocr)

    inference_time = time.time()
    print(f"Time for inference: {inference_time - ocr_time:.2f}s")
        
    if prediction is not None:    
        # Process and check prediction results
        dict_pred = dict(zip(COLUMNS, prediction))
        dicts_check = check_from_config(dict_pred, path_config)
        table = create_result_table(COLUMNS, dict_pred, dicts_check, path_config)
        
        check_time = time.time()
        print(f"Time for checking and save result: {check_time - inference_time:.2f}s")
        
        print(table)
    else:
        print("No prediction available")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Predict Estimate")
    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")
    args = parser.parse_args()
    
    # Run the client pipeline
    client_pipeline(args.path_config)