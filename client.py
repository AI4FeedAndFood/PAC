import requests
import argparse
from call_ocr import from_config_to_text
from constant import COLUMNS
import pandas as pd 
from controle import check_from_config, create_result_table
import time 

def generate_text_from_api(text_ocr, api_url="http://127.0.0.1:8000/generate_text/"):
    data = { "text_ocr": text_ocr}
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
def client_pipeline(path_config):
    starting_time = time.time()
    
    text_ocr = from_config_to_text(path_config)
    
    ocr_time = time.time()    
    print(f"Time for calling OCR: {ocr_time - starting_time}")

    prediction = generate_text_from_api( text_ocr)

    inference_time = time.time()
    print(f"Time for inference: {inference_time - ocr_time}")
        
    if prediction != None:    
        dic_pred =  dict(zip(COLUMNS, prediction))
        df_pred = pd.DataFrame(dic_pred, index=[0])
        dic_check = check_from_config(df_pred, args.path_config)
        table = create_result_table(COLUMNS, dic_pred, dic_check, args.path_config)
        
        check_time = time.time()
        print(f"Time for checking and save result: {check_time - inference_time}")
        
        print(table)
    else:
        print("Aucune prédiction")


if __name__ == "__main__":
    print("Version 3.8")
    
    parser = argparse.ArgumentParser(description="Predict Estimate")
    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")
    args = parser.parse_args()
    
    client_pipeline(args.path_config)