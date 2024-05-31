#!pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
# !pip install --no-deps packaging ninja einops flash-attn trl peft accelerate bitsandbytes
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
import torch
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import math
import argparse

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
          elif len(y[l])>0 and len(y_target[l])>0 and y[l][0] == '<' and  y_target[l][0] == '<':
            if float(y[l][1:]) == float(y_target[l][1:]):
              k+=1
    return k/max(len(y), len(y_target))

def compute_accuracy_only_standard_values(y, y_target):
    k = 0
    n = 13
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
          elif len(y[l])>0 and len(y_target[l])>0 and y[l][0] == '<' and  y_target[l][0] == '<':
            if float(y[l][1:]) == float(y_target[l][1:]):
              k+=1
    return k/n

def compute_extraction_precision(y_pred, y_target):
    n_min = min(len(y_pred), len(y_target))
    n_max = max(len(y_pred), len(y_target))
    k = 0
    for l in range(n_min):
        if (y_pred[l] == '' and y_target[l] == '') or (y_pred[l] != '' and y_target[l] != ''):
            k += 1
    return k/n_max

def compute_value_accuracy(y_pred, y_target):
    n_min = min(len(y_pred), len(y_target))
    n_max = 0
    k = 0
    for l in range(n_min):
        if y_pred[l] != '':
            n_max += 1
            try:
              value = float(y_pred[l])
              value_target = float(y_target[l])
              if value == value_target or (math.isnan(value) and math.isnan(value_target)):
                k+=1
            except:
              if str(y_pred[l]).strip().lower() == str(y_target[l]).strip().lower():
                k+=1
              elif len(y_pred[l])>0 and len(y_target[l])>0 and y_pred[l][0] == '<' and  y_target[l][0] == '<':
                if float(y_pred[l][1:]) == float(y_target[l][1:]):
                  k+=1
    if n_max != 0:
        return k/n_max
    else:
        return 1

def generate(system_prompt, prompt, model, tokenizer, echo = True):
    inputs = tokenizer(
    [
        f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|end_header_id|>"
    ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    if echo:
        return text_output
    else:
        return text_output[0].split('assistant')[-1]

def test_model(model,tokenizer, dataset_test):
    FastLanguageModel.for_inference(model)
    dic_accuracy = {}
    accuracy_tot = 0
    accuracy_std_tot = 0
    extraction_precision_tot = 0
    value_accuracy_tot = 0
    n = len(dataset_test)
    print("Start computing accuracy:")
    for k in range(n):
        print(f"{k}/{n}")
        pred = generate(dataset_test["instruction"][k],dataset_test["input"][k], model,tokenizer, echo = False )
        accuracy = compute_accuracy(pred.split(','), dataset_test["output"][k].split(','))
        accuracy_std = compute_accuracy_only_standard_values(pred.split(','), dataset_test["output"][k].split(','))
        extraction_precision = compute_extraction_precision(pred.split(','), dataset_test["output"][k].split(','))
        value_accuracy = compute_value_accuracy(pred.split(','), dataset_test["output"][k].split(','))

        accuracy_tot += accuracy/n
        accuracy_std_tot += accuracy_std/n
        extraction_precision_tot += extraction_precision/n
        value_accuracy_tot += value_accuracy/n
    dic_accuracy['accuracy_tot'] = accuracy_tot
    dic_accuracy['accuracy_std_tot'] = accuracy_std_tot
    dic_accuracy['extraction_precision_tot'] = extraction_precision_tot
    dic_accuracy['value_accuracy_tot'] = value_accuracy_tot
    return dic_accuracy


def finetuning(path_config):
    
    with open(path_config, 'r') as f:
        config = json.load(f)
    print('###START FINETUNING###')
    print()
    print(f"Config is loaded from {path_config}")
    for conf  in config:
        print(conf)
        
    save_on_disk = config.get("model_config").get("save_on_dick")
    push_on_hub = config.get("model_config").get("push_on_hub")
    new_model = push_on_hub = config.get("model_config").get("new_model")

    if not save_on_disk and not push_on_hub:
        print()
        print(f"Warning: you will not save or push your model. CRTL + Z to abort ?")

    dataset_train_name = config.get("datasets").get("dataset_train_name")
    dataset_train = load_dataset(dataset_train_name)[config.get("datasets").get("split")]
    print()
    print(f"Dataset_train is loaded from {dataset_train_name}")
    
    dataset_test_name = config.get("datasets").get("dataset_test_name")
    dataset_test = load_dataset(dataset_test_name)[config.get("datasets").get("split")]
    print(f"Dataset_test is loaded from {dataset_test_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.get("model_config").get("base_model"),
        max_seq_length = config.get("model_config").get("max_seq_length"),
        dtype = config.get("model_config").get("dtype"),
        load_in_4bit = config.get("model_config").get("load_in_4bit"))

    if new_model:
        model = FastLanguageModel.get_peft_model(
            model,
            r = config.get("lora_config").get("r"),
            target_modules = config.get("lora_config").get("target_modules"),
            lora_alpha = config.get("lora_config").get("lora_alpha"),
            lora_dropout = config.get("lora_config").get("lora_dropout"),
            bias = config.get("lora_config").get("bias"),
            use_gradient_checkpointing = config.get("lora_config").get("use_gradient_checkpointing"),
            random_state = 42,
            use_rslora = config.get("lora_config").get("use_rslora"),
            use_dora = config.get("lora_config").get("use_dora"),
            loftq_config = config.get("lora_config").get("loftq_config"),
        )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,

        train_dataset = dataset_train,

        dataset_text_field = config.get("datasets").get("input_field"),
        max_seq_length = config.get("model_config").get("max_seq_length"),
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config.get("training_config").get("per_device_train_batch_size"),
            gradient_accumulation_steps = config.get("training_config").get("gradient_accumulation_steps"),
            warmup_steps = config.get("training_config").get("warmup_steps"),
            max_steps = config.get("training_config").get("max_steps"),
            num_train_epochs= config.get("training_config").get("num_train_epochs"),
            learning_rate = config.get("training_config").get("learning_rate"),
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = config.get("training_config").get("logging_steps"),
            optim = config.get("training_config").get("optim"),
            weight_decay = config.get("training_config").get("weight_decay"),
            lr_scheduler_type = config.get("training_config").get("lr_scheduler_type"),
            seed = 42,
            output_dir = config.get("training_config").get("output_dir"),
        ),
    )

    test_log_old_model = test_model(trainer.model,trainer.tokenizer, dataset_test )
    print()
    print(f"Old accuracies are:")
    print(test_log_old_model)
    torch.cuda.empty_cache()
    FastLanguageModel.for_training(model)
    trainer_stats = trainer.train()
    print()
    print(f"{trainer_stats=}")
    test_log_new_model = test_model(trainer.model,trainer.tokenizer,dataset_test )
    print()
    print(f"New accuracies are:")
    print(test_log_new_model)

    save_model_path = config.get("model_config").get("save_model_path")

    if save_on_disk:
        trainer.tokenizer.save_pretrained(save_model_path)
    elif push_on_hub:
        trainer.model.save_pretrained(save_model_path)
    else: 
        print()
        print(f"Warning: you don't save or push your model.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Finetuning model")

    parser.add_argument("config", type=str, help="Path for config.json")

    args = parser.parse_args()
    finetuning(args.config)