import pandas as pd
import argparse

#!python /content/extract_NV_from_excel.py "/content/drive/MyDrive/Data/set4-pdf/extraction-set-4.xlsx" "/content/drive/MyDrive/Data/set4-pdf/datas.csv" 0 --output "/content/here.csv" 

parameter_names = ['Energy value (kJ)','Energy value (kcal)',
                  'Fat','saturated fatty acids in the product',  'monounsaturated fatty acids in the product','polyunsaturated fatty acids in the product',
                  'Carbohydrates (available)','Sum of sugars','Sugar ( soluble )','Polyol sum', 'Starch',
                  'Total dietary fibre', 'Protein', 'Salt (NaCl) ex Na',
                  'Vitamin A','Vitamin D3','Vitamin E (alpha tocopheryl acetate)', 'Vitamin K1', 'Vitamin C',
                  'Vitamin B1 thiamin base', 'Vitamin B2 (riboflavin)','Niacin (Vitamin PP or B3)', 'Vitamin B6 (pyridoxine)', 'Total Folate (vitamin B9)','Vitamin B12 (cyanocobalamin)',
                  'Biotin', 'D-Pantothenic acid (Vitamin B5)',
                  "Potassium (K)", 'Chlorides (Cl)','Calcium (Ca)', 'Phosphorus', 'Magnesium (Mg)',
                  'Iron (Fe)', 'Zinc (Zn)','Copper (Cu)','Manganese (Mn)', 'Fluor ??' , 'Selenium (Se)', 'Chromium (Cr)', 'Molybdenum (Mo)' ,'Iodine (I)']

codes_package_nut = ['PAA9E', 'PAA9A', 'PAA9C', 'PAA9D', 'PVA5F', 'PAA5J', 'PVA6R', 'PAA7J',
       'PAA9G', 'PY59U', 'PAA9M', 'PVAL2', 'PAA42', 'PAA29', 'PAA3Y', 'PAA43',
       'PAA3K', 'PAA9I', 'PAA45', 'PAA9H', 'PY57Y', 'PAAPL', 'PAA3N', 'PAA5L',
       'PAA7E', 'PAA5K', 'PY5LK', 'PAA5Y', 'PY5AM', 'PAA7G', 'PY504', 'PY59T',
       'PY59K', 'PY59S', 'PVA1Z', 'PVA5X', 'PVAN9', 'PAAFY', 'PAA8G', 'PY5F9',
       'PAA3C', 'PAA9K', 'PAA9J', 'PVA2A', 'PY56V', 'PY56Z', 'PAAMG', 'PVA2X',
       'PVA7M', 'PY52Z', 'PAAFR', 'PY56J', 'PVA2F', 'PY52H', 'PAA62', 'PVA5Z',
       'PVA1V', 'PVA1W', 'PAA5Q', 'PAA5R', 'PAA5S', 'PAA9F', 'PAA7W', 'PAA8C',
       'PY57U', 'PY5DR', 'PVA6N', 'PAAGU', 'PAA3P', 'PAA9O', 'PAA2O', 'PVA2L',
       'PY52G', 'PFU02', 'PAAP3']






def process_data(path_extraction, path_sample_code_path, sheet_id):
    # Lecture des données
    print(f'Reading excel {path_extraction}...')
    df_extraction = pd.read_excel(path_extraction, sheet_name=sheet_id)
    print(f'Reading csv {path_sample_code_path}')
    df_sample_code_path = pd.read_csv(path_sample_code_path)
    print(f'Reading complished')

    print(f"Filtering the datas...")
    # Filtrage initial
    mask = df_extraction["PackageCode"].isin(codes_package_nut) & df_extraction["PackageCode"].notna()
    df_temp = df_extraction[mask]
    sample_codes = list(df_temp["samplecode"].value_counts().index)

    # Filtrage principal
    mask = (df_extraction["OriginalFractionCode"] == 'J01') & \
           (df_extraction["EstimatedValue"].notna()) & \
           (df_extraction["samplecode"].isin(sample_codes)) & \
           (df_extraction["ParameterName"].isin(parameter_names))
    df_extraction_filtered = df_extraction[mask]

    # Création du DataFrame pivot
    sample_codes = df_extraction_filtered["samplecode"].unique()
    dic = {parameter: [None] * len(sample_codes) for parameter in parameter_names}
    dic["samplecode"] = sample_codes

    dict_sample_path = dict(zip(df_sample_code_path['sampleCodes'], df_sample_code_path['path']))
    dic["path"] = [dict_sample_path[sample_code] for sample_code in dic["samplecode"]]

    print(f"Filtering complished")
    print(f"Creating table: sample code/nutritional values...")
    df_pivot = pd.DataFrame(dic)
    
    # Réorganisation des colonnes
    df_pivot = df_pivot[["path", "samplecode"] + parameter_names]

    # Remplissage des valeurs
    for index, row in df_extraction_filtered.iterrows():
        samplecode = row["samplecode"]
        df_pivot.loc[df_pivot["samplecode"] == samplecode, row["ParameterName"]] = row["EstimatedValue"]
        df_pivot.loc[df_pivot["samplecode"] == samplecode, "PackageCode"] = row["PackageCode"]

    # Ajustement final
    df_pivot['Sum of sugars'] = df_pivot['Sum of sugars'].fillna(df_pivot['Sugar ( soluble )'])
    print(f"Table created")

    return df_pivot

def main():
    parser = argparse.ArgumentParser(description="Process extraction data and generate CSV.")
    parser.add_argument("path_extraction", help="Path to the extraction Excel file")
    parser.add_argument("path_sample_code_path", help="Path to the sample code CSV file")
    parser.add_argument("sheet_id", type=int, help="Sheet ID in the extraction Excel file")
    parser.add_argument("--output", default="extracted_data_original.csv", help="Output CSV file name")

    args = parser.parse_args()

    df_result = process_data(args.path_extraction, args.path_sample_code_path, args.sheet_id)
    print("Saving the table...")
    df_result.to_csv(args.output, index=False)    
    print(f"Data processed and saved to {args.output}")

if __name__ == "__main__":
    main()