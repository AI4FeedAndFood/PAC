import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
import math
import time 

COLUMN_NAMES = ['Energie(kJ)', 'Energie(kcal)', 'Mat grasse(g)', 'Ac gras sat(g)',
       'Ac gras mono-insat(g)', 'Ac gras polyinsat(g)', 'Glucide(g)',
       'Sucre(g)', 'Polyols(g)', 'Amidon(g)', 'Fibre(g)', 'Proteine(g)',
       'Sel(g)', 'Vit A(µg)', 'Vit D(µg)', 'Vit E(mg)', 'Vit K(µg)',
       'Vit C(mg)', 'Thiamine(mg)', 'Riboflavine(mg)', 'Niacine(mg)',
       'Vit B6(mg)', 'Ac folique(µg)', 'Vit B12(µg)', 'Biotine(µg)',
       'Ac panto(mg)', 'Potassium(mg)', 'Chlorure(mg)', 'Calcium(mg)',
       'Phosphore(mg)', 'Magnesium(mg)', 'Fer(mg)', 'Zinc(mg)', 'Cuivre(mg)',
       'Manganèse(mg)', 'Fluorure(mg)', 'Selenium(µg)', 'Chrome(µg)',
       'Molybdene(µg)', 'Iode(µg)']

def compute_accuracy(y, y_target):
    """
    This function computes the accuracy between two lists: y and y_target.
    It checks if the corresponding elements in the lists are equal, or if both are NaN.
    If the elements are strings, it checks if they are equal after stripping and converting to lower case.
    If the elements are strings starting with '<', it checks if the floating point values after the '<' are equal.

    Parameters:
    y (list): List of predicted values.
    y_target (list): List of target values.

    Returns:
    float: The accuracy as a fraction of the maximum length of y and y_target.
    """
    k = 0
    if len(y) != len(y_target):
      print(f"Taille vecteur sortie {len(y)}, attendu: {len(y_target)}")
    n = min(len(y), len(y_target))
    for l in range(n):
        ##Pre-Process
        try:
          value = float(y[l])
          value_target = float(y_target[l])
          if value == value_target or (math.isnan(value) and math.isnan(value_target)):
            k+=1
        except:
          if str(y[l]).strip().lower() == str(y_target[l]).strip().lower():
            k+=1
          if len(y[l])>0 and len(y_target[l])>0 and y[l][0] == '<' and  y_target[l][0] == '<':
            if float(y[l][1:]) == float(y_target[l][1:]):
              k+=1
    return k/max(len(y), len(y_target))

def load_llm(model_path):
    """
    This function loads a language model and its tokenizer from a given model path.

    Parameters:
    model_path (str): Path to the model.

    Returns:
    tuple: A tuple containing the tokenizer and the model.
    """
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path)

    return tokenizer, model

def load_tokenizer(model_path):
    """
    This function loads a tokenizer from a given model path and adds a special padding token.

    Parameters:
    model_path (str): Path to the model.

    Returns:
    AutoTokenizer: The tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'
    return tokenizer

def load_model(model_path):
    """
    This function loads a quantized language model from a given model path.

    Parameters:
    model_path (str): Path to the model.

    Returns:
    AutoModelForCausalLM: The language model.
    """
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 )#bnb_4bit_compute_dtype=torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto")

    return model

def create_prompt(ocr_text, pre_prompt_dic):
    """
    This function creates a list of messages for a chatbot, using a dictionary of pre-prompts and the OCR text.

    Parameters:
    ocr_text (str): The OCR text.
    pre_prompt_dic (dict): A dictionary containing various pre-prompts.

    Returns:
    list: A list of messages for a chatbot.
    """
    pre_prompt_goal = pre_prompt_dic["pre_prompt_goal"]
    pre_prompt_format= pre_prompt_dic["pre_prompt_format"]
    pre_prompt_tips= pre_prompt_dic["pre_prompt_tips"]
    pre_prompt_indication= pre_prompt_dic["pre_prompt_indication"]
    pre_prompt_ocr= pre_prompt_dic["pre_prompt_ocr"]
    pre_answer= pre_prompt_dic["pre_answer"]
    messages = [
        {"role": "user", "content":pre_prompt_goal + pre_prompt_format + pre_prompt_tips + pre_prompt_indication + pre_prompt_ocr},
        {"role": "assistant", "content": pre_answer},
        {"role": "user", "content": ocr_text}]
    return messages


def inference(model, tokenizer, messages, max_new_tokens = 100, temp = 0.1,do_sample = False, device =  "cuda"):
    """
    This function generates a response from a language model, given a list of messages.

    Parameters:
    model: The language model.
    tokenizer: The tokenizer for the model.
    messages (list): A list of messages for a chatbot.
    max_new_tokens (int, optional): The maximum number of new tokens to generate. Default is 100.
    temp (float, optional): The temperature for sampling. Default is 0.1.
    do_sample (bool, optional): Whether to use sampling. Default is False.
    device (str, optional): The device to use for computation. Default is "cuda".

    Returns:
    str: The generated response.
    """
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens,do_sample = do_sample, temperature = temp)
    output_txt = tokenizer.decode(outputs[0][-max_new_tokens:], skip_special_tokens=True)
    return output_txt

def convertion_result_LLM_to_csv(output_txt):
    """
    This function converts the output of a language model into a list of strings.

    Parameters:
    output_txt (str): The output of a language model.

    Returns:
    list: A list of strings.
    """
    try:
        output_txt = output_txt.split(']')[1]
    except:
        output_txt = output_txt
    output_txt = output_txt.split('\n')[0]
    y = output_txt.split(',')
    return y

def extract_info_with_llm_from_ocr_text(txt_ocr, path_llm, pre_prompt_dic, max_new_tokens = 100, temp = 0.1,do_sample = False, device =  "cuda"):
    """
    This function extracts information from OCR text using a language model and a dictionary of pre-prompts.

    Parameters:
    txt_ocr (str): The OCR text.
    path_llm (str): Path to the language model.
    pre_prompt_dic (dict): A dictionary containing various pre-prompts.
    max_new_tokens (int, optional): The maximum number of new tokens to generate. Default is 100.
    temp (float, optional): The temperature for sampling. Default is 0.1.
    do_sample (bool, optional): Whether to use sampling. Default is False.
    device (str, optional): The device to use for computation. Default is "cuda".

    Returns:
    list: A list of strings extracted from the OCR text.
    """
    starting_time = time.time()
    message = create_prompt(txt_ocr,pre_prompt_dic)
    tokenizer, model = load_llm(path_llm)
    loading_time = time.time()
    print(f"Loading succed in {loading_time - starting_time}s.")
    output = inference(model, tokenizer, message, max_new_tokens, temp,do_sample, device)
    output = convertion_result_LLM_to_csv(output)
    print(f"Inference in {time.time() - starting_time}s]")
    return output

def extract_info_with_llm_from_ocr_text_in_pandas(df,column_ocr_name, path_llm, pre_prompt_dic, max_new_tokens = 100, temp = 0.1,do_sample = False, device =  "cuda"):
    """
    This function extracts information from OCR text in a pandas DataFrame using a language model and a dictionary of pre-prompts.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the OCR text.
    column_ocr_name (str): The name of the column containing the OCR text.
    path_llm (str): Path to the language model.
    pre_prompt_dic (dict): A dictionary containing various pre-prompts.
    max_new_tokens (int, optional): The maximum number of new tokens to generate. Default is 100.
    temp (float, optional): The temperature for sampling. Default is 0.1.
    do_sample (bool, optional): Whether to use sampling. Default is False.
    device (str, optional): The device to use for computation. Default is "cuda".

    Returns:
    DataFrame: A pandas DataFrame containing the extracted information.
    """
    starting_time = time.time()
    preds = []
    txts = df[column_ocr_name]
    tokenizer, model = load_llm(path_llm)
    loading_time = time.time()
    print(f"Loading succed in {loading_time - starting_time}s.")
    n = len(txts)
    k = 0 
    for txt in txts:
       time_before = time.time()
       message = create_prompt(txt,pre_prompt_dic)
       output = inference(model, tokenizer, message, max_new_tokens, temp,do_sample, device)
       preds.append(convertion_result_LLM_to_csv(output))
       time_after = time.time()
       k +=1
       print(f"{k}/{n}, [{time_after - time_before}s]")

    df_pred = pd.DataFrame(columns=COLUMN_NAMES)


    for output in preds:
        if len(output)>len(COLUMN_NAMES):
            output = output[:len(COLUMN_NAMES)]
            try:
               print(f"Warning: {df['path']} result get cutted because lenght > ")
            except:
               print(f"Warning: result get cutted because lenght > ")
        elif len(output)<len(COLUMN_NAMES):
            output += [''] * (len(COLUMN_NAMES) - len(output))
            try:
               print(f"Warning: {df['path']} result get padded because lenght < ")
            except:
               print(f"Warning: result get padded because lenght < ")

        df_pred = pd.concat([df_pred, pd.DataFrame([output], columns=COLUMN_NAMES)], ignore_index=True)
        df_pred['path'] = df["path"]
    return df_pred


pre_prompt_dic = {
    'pre_prompt_goal' : "Tu es une IA qui doit convertir un texte obtenu via un OCR, en un tableau csv regroupant les différentes informations nutritionelles pour 100g ou 100ml. Ta réponse ne contiendra que le tableau csv.\n",
    "pre_prompt_format" : "Le format csv attendue est 1 ligne, 40 colonnes. Les colonnes sont: 'Energie(kJ),Energie(kcal),Mat grasse(g),Ac gras sat(g),Ac gras mono-insat(g),Ac gras polyinsat(g),Glucide(g),Sucre(g),Polyols(g),Amidon(g),Fibre(g),Proteine(g),Sel(g),Vit A(µg),Vit D(µg),Vit E(mg),Vit K(µg),Vit C(mg),Thiamine(mg),Riboflavine(mg),Niacine(mg),Vit B6(mg),Ac folique(µg),Vit B12(µg),Biotine(µg),Ac panto(mg),Potassium(mg),Chlorure(mg),Calcium(mg),Phosphore(mg),Magnesium(mg),Fer(mg),Zinc(mg),Cuivre(mg),Manganèse(mg),Fluorure(mg),Selenium(µg),Chrome(µg),Molybdene(µg),Iode(µg)\n",
    "pre_prompt_tips" : "Attention, certaines colonnes seront parfois vide.\n",
    "pre_prompt_indication": "Peux -tu convertir le texte suivant:\n",
    "pre_prompt_ocr" : "Emincé de poulet sur lit de pâtes complètes, légumes du sud et parmesan Riche en fibres. Informations nutritionnelles : Pour 100g Pour 1 part Pour 1 part (380g) % des AR* Energie 462KJ 1756KJ 21% 110kcal 418kcal Matières grasses 2,3g 8,8g 13% dont acides gras saturés 0,8g 3,0g 15% Glucides 12,8g 48,6g 19% dont sucres 3,0g 11,4g 13% Fibres alimentaires 6,4g 24,3g 1 Protéines 7,9g 30g 60% Sel 0,4g 1,5g 25% apport de référence pour un adulte-type 8400KJ/2000kcal. Pour 1 part (380g) : 1756KJ mat. gr saturés sucres sel 418kcal 8,8g 3,0g 11,4g 1,5g 21% 13%* 15%* 13%* 25%*. Pour 100g : 462KJ/110kcal. Mar *apport de référence pour un adulte-type (8400KJ/2000kcal). que Poids net : 380g : 1 part PLUS Service Information Conso mmateur Marque PLus BP 71-75 116 PARIS cedex.",
    "pre_answer" : "462,110, 2.3,0.8,,,12.8,3,,,6.4,7.9,0.4,,,,,,,,,,,,,,,,,,,,,,,,,\n"}