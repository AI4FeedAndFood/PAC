import time
from read_config import read_config_predict
from unsloth import FastLanguageModel
import argparse
import torch #for eval(config.get("dtype")) !
import call_ocr as co
from controle import check_from_config, create_result_table
from constant import COLUMNS, INSTRUCTION
from mesure import test_datas
import pandas as pd

def split_into_list(pred, len_target = -1):
    """
    Splits the prediction string into a list of individual predictions.

    Parameters:
    - pred (str): The prediction string to be split.
    - len_target (int, optional): The target length of the list. If not specified, it defaults to the length of COLUMNS.

    Returns:
    - list: A list of predictions with a length equal to len_target.
    """
    list_pred = pred.split(',')
    if len_target == -1:
        len_target = len(COLUMNS)

    if  len_target > len(list_pred):
        list_pred.extend(['' for _ in range(len_target - len(list_pred))])

    return list_pred[0: len_target]
    

def generate(system_prompt, input, model, tokenizer, echo = True):
    """
    Generates a response using the language model.

    Parameters:
    - system_prompt (str): The system prompt to guide the model.
    - input (str): The input text for the model.
    - model: The pre-trained language model.
    - tokenizer: The tokenizer associated with the model.
    - echo (bool, optional): Whether to echo the entire response or just the assistant's part. Defaults to True.

    Returns:
    - str or None: The generated text or None if an error occurs.
    """
    try:
        text_prompt = [f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{input}<|end_header_id|>"]
        
        prompts = tokenizer(text_prompt, return_tensors = "pt").to("cuda")

        outputs = model.generate(**prompts, max_new_tokens = 256, pad_token_id=tokenizer.eos_token_id)#, use_cache = False) <-- create some bugs 
        
        generated_tokens = outputs[:, prompts["input_ids"].shape[-1]:]
        
        text_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        if echo:
            return text_output
        else:
            #usual case
            return text_output[0].split('assistant')[-1]
    except Exception as e:
        print(f"An error occured in generate function : {e}")
        return None

def from_config_load_model(path_config):
    """
    Loads the model and tokenizer based on the configuration file.

    Parameters:
    - path_config (str): The path to the configuration file.

    Returns:
    - tuple: The loaded model and tokenizer.
    """
    config = read_config_predict(path_config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.get("model"),
        max_seq_length = config.get("max_seq_length"),
        dtype = eval(config.get("dtype")),
        load_in_4bit = config.get("load_in_4bit"))
    
    return model, tokenizer


def predict_estimates(path_config, scan_path = None, model = None, tokenizer = None):
    """
    Predicts estimates based on the OCR results and the model.

    Parameters:
    - path_config (str): The path to the configuration file.
    - scan_path (str, optional): The path to the scan for OCR. Defaults to None.
    - model: The pre-trained language model. Defaults to None.
    - tokenizer: The tokenizer associated with the model. Defaults to None.

    Returns:
    - dict: A dictionary mapping columns to their predicted values.
    """
    start_time = time.time()
    if scan_path != None:
        config = read_config_predict(path_config)
        text_ocr = co.from_path_to_text_OCRAzure(scan_path, model_yolo_path = config.get("model_yolo"))
    else:
        text_ocr = co.from_config_to_text(path_config)
    ocr_time = time.time()
    print(f"OCR complished in {ocr_time - start_time}")
    print(f"OCR prediction: \n {text_ocr}\n")
    if model == None or tokenizer == None:
        model, tokenizer = from_config_load_model(path_config)
    FastLanguageModel.for_inference(model)
    load_llm_time = time.time()
    print(f"LLM loaded in {load_llm_time - ocr_time}")

    pred = generate(INSTRUCTION, text_ocr, model, tokenizer, echo = False)
    if pred is None:
        print("Function generate return None. Treatment is stopped.")
        return None
    llm_time = time.time()
    print(f"Inference in {llm_time - load_llm_time}")

    list_pred = split_into_list(pred)
    print(list_pred)
    k = 0
    while  k <len(list_pred) or k < len(COLUMNS):

        if list_pred[k] != "":
            print("\n")

            print(f"{COLUMNS[k]}: {list_pred[k]}")
        k += 1
    return dict(zip(COLUMNS, list_pred))

def test_pipeline(path_config, path_dataset_test, path_csv, split = "Test"):
    """
    Tests the entire prediction pipeline on a test dataset.

    Parameters:
    - path_config (str): The path to the configuration file.
    - path_dataset_test (str): The path to the test dataset.
    - path_csv (str): The path to save the CSV file with the results.
    - split (str, optional): The dataset split to use. Defaults to "Test".

    Returns:
    - None
    """
    from datasets import load_from_disk
    
    if split == None or split.lower() == "none":
        dataset_test = load_from_disk(path_dataset_test)
    else:
        dataset_test = load_from_disk(path_dataset_test)[split]
    scan_paths = dataset_test["path"]
    outputs =  dataset_test["output"]
    model, tokenizer = from_config_load_model(path_config)
    FastLanguageModel.for_inference(model)
    y_preds = []
    y_targets = []
    
    y_preds_TD = []
    y_targets_TD = []
    
    y_preds_not_TD = []
    y_targets_not_TD = []
    
    estimate_dict = {column: [] for column in COLUMNS}
    estimate_dict['path'] = []
    estimate_dict['TD'] = []
    k = 0
    n = len(scan_paths)
    config = read_config_predict(path_config)
    for scan_path, output in zip(scan_paths,outputs):
        print(f"{k}/{n}")
        k+=1
        text_ocr, is_crop = co.from_path_to_text_OCRAzure(scan_path, model_yolo_path = config.get("model_yolo"),bool_is_crop = True)

        pred = generate(INSTRUCTION, text_ocr, model, tokenizer, echo = False)
        y_pred = split_into_list(pred)
        
        y_target = split_into_list(output)
        
        y_preds.append(y_pred)
        y_targets.append(y_target)
        
        if is_crop:
            y_preds_TD.append(y_pred)
            y_targets_TD.append(y_target)
        else:
            y_preds_not_TD.append(y_pred)
            y_targets_not_TD.append(y_target)
            
        dict_pred = dict(zip(COLUMNS, y_pred))
        for column in COLUMNS:
            if column in dict_pred.keys():
                estimate_dict[column].append(dict_pred[column])
            else:
                estimate_dict[column].append("")
                
        estimate_dict['path'].append(scan_path)
        estimate_dict['TD'].append(is_crop)
    df = pd.DataFrame(estimate_dict)
    df.to_csv(path_csv)
    
    
    
    dic_accuracy = test_datas(y_preds, y_targets)
    dic_accuracy_TD = test_datas(y_preds_TD, y_targets_TD)
    dic_accuracy_not_TD = test_datas(y_preds_not_TD, y_targets_not_TD)
    print(f"{dic_accuracy=}")
    print(f"{dic_accuracy_TD=}")
    print(f"{dic_accuracy_not_TD=}")
    
        

def lunch_pipeline(path_config):
    """
    Launches the model prediction estimates pipeline.

    Parameters:
    - path_config (str): The path to the configuration file.

    Returns:
    - None
    """
    print("### Lunch Model Prediction Estimates ###")
    
    dic_pred = predict_estimates(path_config)
    
    if dic_pred != None:
        dicts_check = check_from_config(dic_pred, path_config)
        df_result = create_result_table(COLUMNS, dic_pred, dicts_check, path_config)
        print(df_result)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict Estimate")

    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")

    args = parser.parse_args()
    lunch_pipeline(args.path_config) 
    
    
