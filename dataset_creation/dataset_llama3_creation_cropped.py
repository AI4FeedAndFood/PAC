import argparse
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from crop_with_yolo import crop_with_yolo
import call_ocr
from time import time
import os

def load_dataset_from_disk(dataset_path):
    return load_from_disk(dataset_path)

def process_image(old_path, old_input, output, output_dir, yolo_model_path):
    filename = old_path.split('/')[-1].split("_")[0]
    cropped_imgs, pages = crop_with_yolo(old_path, yolo_model_path)
    
    if len(cropped_imgs) > 0:
        text_ocr = ""
        new_path = ""
        for k, cropped_img in enumerate(cropped_imgs):
            new_path = f"{output_dir}{filename}_cropped_{k}.png"
            cropped_img.save(new_path)
            text_ocr += call_ocr.from_path_to_text_OCRAzure(new_path)
        
        return [(new_path, text_ocr, output, True), (old_path, old_input, output, False)]
    else:
        return [(old_path, old_input, output, False)]

def create_new_dataset(processed_data, instruction):
    new_paths, new_inputs, new_outputs, TDs = zip(*processed_data)
    
    new_prompts = [f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{output}<|eot_id|>""" for input, output in zip(new_inputs, new_outputs)]
    
    new_dict = {
        'path': new_paths,
        'output': new_outputs,
        'instruction': [instruction] * len(new_paths),
        'input': new_inputs,
        'prompt': new_prompts,
        'TD' : TDs 
    }
    
    return Dataset.from_dict(new_dict)

def process(args, dataset):
    old_paths = dataset['path']
    instruction = dataset['instruction'][0]
    outputs = dataset['output']
    old_inputs = dataset['input']
    
    processed_data = []
    n = len(old_paths)
    tot_time = 0
    t = time()
    
    for l, (old_path, old_input, output) in enumerate(zip(old_paths, old_inputs, outputs), 1):
        tot_time += (time() - t)
        t = time()
        remaining_time = (n - l) * tot_time / (l + 0.01)
        print(f"{l}/{n} -- {tot_time:.1f}/{remaining_time:.1f}")
        
        processed_data.extend(process_image(old_path, old_input, output, args.output_dir, args.yolo_model))
    new_dataset = create_new_dataset(processed_data, instruction)
    new_dataset = new_dataset.shuffle()
    return new_dataset
    
def main(args):
    split = args.split
    dataset = load_dataset_from_disk(args.input_dataset)[split]
    
    print(f"Process dataset train:")
    dataset_cropped = process(args, dataset)
   
    combined_dataset = DatasetDict({
    split: dataset_cropped
    })
    combined_dataset.save_to_disk(args.output_dataset)
    print(f"Dataset saved to {args.output_dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with table detection and OCR")
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--output_dataset", type=str, required=True, help="Path to save the output dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cropped images")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--split", type=str, required=True, help="Split name")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)