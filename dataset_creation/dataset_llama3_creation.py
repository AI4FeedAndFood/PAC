### Llama 3 format prompt (+ path PDF, output)



import pandas as pd
from datasets import Dataset, DatasetDict
import argparse
from constant import OCR_COLUMN, INSTRUCTION

#INPUT_DATASET = '/content/drive/MyDrive/Data/set-2-pdf-quatro/datas.csv'
#OUTPOUT_DATASET_INSTRUCT = "/content/drive/MyDrive/Data/set-2-pdf-quatro/dataset llama"


def df_to_csv_nutrition(df,start_id, lenght_nutrition_table = 40):
    """
    This function converts a specified range of columns from a DataFrame to a CSV format.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.
    start_id (int): The starting column index to convert.
    lenght_nutrition_table (int, optional): The number of columns to convert. Defaults to 40.

    Returns:
    list: A list of strings where each string is a row in the CSV format.
    """
    return df[df.columns[start_id:start_id + lenght_nutrition_table]].to_csv(index=False).split("\n")

def create_dataset_llama3(input_csv, output_dataset, id_starting_estimate_column = 4,lenght_nutrition_table = 40, save_to_disk = True, push_to_hub = False, tokens_for_transformer = ""):
    """
    This function creates a dataset from a CSV file and saves it to disk or pushes it to a hub.
    Warning: Dataset format must be "Energie(kJ),Energie(kcal),Mat grasse(g),Ac gras sat(g),Ac gras mono-insat(g),Ac gras polyinsat(g),Glucide(g),Sucre(g),Polyols(g),Amidon(g),Fibre(g),Proteine(g),Sel(g),Vit A(µg),Vit D(µg),Vit E(mg),Vit K(µg),Vit C(mg),Thiamine(mg),Riboflavine(mg),Niacine(mg),Vit B6(mg),Ac folique(µg),Vit B12(µg),Biotine(µg),Ac panto(mg),Potassium(mg),Chlorure(mg),Calcium(mg),Phosphore(mg),Magnesium(mg),Fer(mg),Zinc(mg),Cuivre(mg),Manganèse(mg),Fluorure(mg),Selenium(µg),Chrome(µg),Molybdene(µg),Iode(µg)"
    API OCR Azure is called. 
    If the dataset already has a column name "Azure_text" (check the constant OCR_COLUMN), the API OCR will not be called. 
    
    Parameters:
    input_csv (str): The path to the input CSV file.
    output_dataset (str): The name of the output dataset.
    id_starting_estimate_column (int, optional): The starting column index for Estimates. Defaults to 4.
    lenght_nutrition_table (int, optional): The number of columns for the nutrition table. Defaults to 40.
    save_to_disk (bool, optional): If True, the dataset is saved to disk. Defaults to True.
    push_to_hub (bool, optional): If True, the dataset is pushed to a hub. Defaults to False.
    tokens_for_transformer (str, optional): If you want to upload your dataset on hub, pls enter your tokens
    Returns:
    None
    """    
    print(f"Starting dataset creation")
    df_input = pd.read_csv(input_csv)
    outputs = df_to_csv_nutrition(df_input, start_id = id_starting_estimate_column, lenght_nutrition_table = lenght_nutrition_table)[1:-1]
    print(f'CSV reading completed')
    if OCR_COLUMN in df_input.columns:
        inputs =  df_input[OCR_COLUMN].to_list()
    else:
        import call_ocr as co ## import here to resolve dependencies (better for Collab use case, it avoid to download OCR packages)
        co.add_text_OCRAzure_to_dataset(df_input)
        inputs =  df_input[OCR_COLUMN].to_list()
        
    paths = df_input["path"].to_list()

    prompts = []
    for k in range(len(outputs)):
        input = inputs[k]
        output = outputs[k]
        prompts.append(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{output}<|eot_id|>""")

    df = pd.DataFrame({'prompt': prompts,
                    'input' : inputs,
                    'output' : outputs,
                    'path' : paths,
                    'instruction': INSTRUCTION})

    df.reset_index(drop=True, inplace=True)
    llama3_dataset = DatasetDict({"train": Dataset.from_pandas(df)})

    if save_to_disk:
        llama3_dataset.save_to_disk(output_dataset)
        print(f'Dataset save in {output_dataset}')
    elif push_to_hub:
        if tokens_for_transformer != "":
            from huggingface_hub import login
            login(tokens_for_transformer)
        llama3_dataset.push_to_hub(output_dataset)
        print(f'Dataset push in {output_dataset}')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Créer un dataset pour Llama 3")

    parser.add_argument("input_csv", type=str, help="Le chemin d'accès au fichier CSV d'entrée")
    parser.add_argument("output_dataset", type=str, help="Le chemin d'accès au fichier de sortie du dataset")
    parser.add_argument("--id_starting_estimate_column", type=int, default=4, help="La colonne de début de l'estimation d'ID (défaut : 4)")
    parser.add_argument("--lenght_nutrition_table", type=int, default=40, help="La longueur de la table de nutrition (défaut : 40)")
    parser.add_argument("--save_to_disk", action="store_true", help="Enregistrer le dataset sur le disque dur")
    parser.add_argument("--push_to_hub", action="store_true", help="Publier le dataset sur le hub")
    parser.add_argument("--tokens_for_transformer", type=str, default="", help="Les jetons pour le transformer (défaut : chaîne vide)")

    args = parser.parse_args()

    create_dataset_llama3(args.input_csv, args.output_dataset, args.id_starting_estimate_column, args.lenght_nutrition_table, args.save_to_disk, args.push_to_hub, args.tokens_for_transformer)