import time
from read_config import read_config_predict
from unsloth import FastLanguageModel
import argparse

import call_ocr as co
from controle import check_from_config, create_result_table
from constant import COLUMNS, INSTRUCTION

def generate(system_prompt, prompt, model, tokenizer, echo = True):
    try:
        inputs = tokenizer(
        [
            f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|end_header_id|>"
        ], return_tensors = "pt").to("cuda")

        n = len(inputs["input_ids"])
        print(f"Len inputs = {n}")
        outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True, pad_token_id=tokenizer.eos_token_id)
        text_output = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        if echo:
            return text_output
        else:
            #print(text_output[0])
            return text_output[0].split('assistant')[-1]
    except Exception as e:
        print(f"An error occured in generete function : {e}")
        return None

def from_config_load_model(path_config):
    config = read_config_predict(path_config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.get("model"),
        max_seq_length = config.get("max_seq_length"),
        dtype = eval(config.get("dtype")),
        load_in_4bit = config.get("load_in_4bit"))
    
    return model, tokenizer


def predict_estimates(path_config):
    start_time = time.time()
    
    text_ocr = co.from_config_to_text(path_config)
    ocr_time = time.time()
    print(f"OCR complished in {ocr_time - start_time}")


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

    list_pred = pred.split(",")
    print(list_pred)
    k = 0
    while  k <len(list_pred) or k < len(COLUMNS):

        if list_pred[k] != "":
            print("\n")

            print(f"{COLUMNS[k]}: {list_pred[k]}")
        k += 1
    return dict(zip(COLUMNS, list_pred))



def lunch_pipeline(path_config):
    print("### Lunch Model Prediction Estimates ###")
    
    list_pred = predict_estimates(path_config)
    
    if list_pred != None:
        list_check = check_from_config(list_pred, path_config)
        df_result = create_result_table(COLUMNS, list_pred, list_check)
        print(df_result)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict Estimate")

    parser.add_argument("path_config", type=str, help="Path for config_predict.json, ex: 'what/is/the/directory/config_predict.json'")

    args = parser.parse_args()
    lunch_pipeline(args.path_config) 
    
    
