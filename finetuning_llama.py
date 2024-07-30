#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
import torch
import json
from unsloth import is_bfloat16_supported
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import argparse
from mesure import test_datas

def test_model(model,tokenizer, dataset_test):
    FastLanguageModel.for_inference(model)
    
    n = len(dataset_test)
    print("Start computing accuracy:")
    y_preds = []
    y_targets = []
    
    for k in range(n):
        print(f"{k}/{n}")
        pred = generate(dataset_test["instruction"][k],dataset_test["input"][k], model,tokenizer, echo = False )
        y_preds.append(pred.split(',')) 
        y_target = dataset_test["output"][k].split(',')
        y_targets.append(y_target)
        
    dic_accuracy = test_datas(y_preds, y_targets)
    
    return dic_accuracy

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
    
def load_dataset_for_finetuning(dataset_name, split, on_disk):    
    if on_disk:
        if split == None or split.lower() == "none":
            dataset = load_from_disk(dataset_name)
        else:
            dataset = load_from_disk(dataset_name)[split]
    else:
        if split == None or split.lower() == "none" :
            dataset = load_dataset(dataset_name)
        else:
            dataset = load_dataset(dataset_name)[split]
        
    print()
    print(f"Dataset is loaded from {dataset_name}")
    return dataset


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

    dataset_train = load_dataset_for_finetuning(config.get("datasets").get("dataset_train_name"), config.get("datasets").get("split_train"), config.get("datasets").get("dataset_train_on_disk"))
    
    dataset_test = load_dataset_for_finetuning(config.get("datasets").get("dataset_test_name"), config.get("datasets").get("split_test"), config.get("datasets").get("dataset_test_on_disk"))
    
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