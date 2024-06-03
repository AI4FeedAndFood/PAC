import pandas as pd 
from constant import PATH_CIQUAL_TABLE,CONVERTER,COLUMNS_FOR_ENERGY,COLUMNS_FOR_CARBOHYDRATE_WEIGHT, COLUMNS_FOR_FAT_WEIGHT, COLUMNS_FOR_TOTAL_WEIGHT,TOTAL_COLUMNS  
import os 
from read_config import read_config_predict
def extract_from_ciqual(df, alim_code, nut_name):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(alim_code, int):
        raise TypeError("alim_code must be a string")
    if not isinstance(nut_name, str):
        raise TypeError("nut_name must be a string")
    if alim_code not in df['alim_code'].to_list():
        raise TypeError("alim_code not in the Ciqual table")
    if nut_name not in df.columns:
        raise TypeError("nut_name not in the Ciqual columns, maibe updtate converter from constant or check Ciqual Table")
    try:
        valeur = df.loc[df['alim_code'] == alim_code, nut_name].values[0]
        return valeur
    except IndexError:
        print( f"Aucune valeur trouvée pour cet alim_code {alim_code} et nut_name {nut_name}.")
        return -1
    
def convert_to_float(value):
    value = value.replace(',', '.')
    characters = ".1234567890"
    try:
        if '-' in value or 'trace' in value or value == 'nan' or value == '':
            return -1
        elif '<' in value:

            value = "".join(x for x in value if x in characters)
            return float(value)
        elif value == "".join(x for x in value if x in characters):
            return float(value)
        else:
          return -1
    except:
          return -2
      
def relative_gap(y_pred, y_target):
    return abs(y_pred - y_target)/(y_target + 0.001)

def is_a_normal_value_for_nut_name(df, y_pred, alim_code, nut_name, max_relative_gab):
    y_target= extract_from_ciqual(df, alim_code, nut_name)
    if y_target == -1:
        return False, 0
    else:
        if isinstance(y_target, str):
            y_target = convert_to_float(y_target)
        if isinstance(y_pred, str):
            y_pred = convert_to_float(y_pred)
    if y_target >= 0 and y_pred >=  0:
        eps = relative_gap(y_pred, y_target)
        if eps < max_relative_gab:
            return True, eps
        else:
            return False, eps
    else:
      return False, 0
  

def check_with_ciqual(df_pred, df_true,alim_code, max_relative_gab, converter):
    check_list = {}
    for column in list(df_pred.columns):

        if column in converter.keys():
              column_ciqual = converter[column]
              is_normal, value = is_a_normal_value_for_nut_name(df_true, df_pred[column].values[0], alim_code, column_ciqual, max_relative_gab)
              value = round(value*100) #convertion en % avec arrondi
              if is_normal:
                  print(f"{column} a une valeur attendue cohérente")
                  check_list[column] =  True
              elif not is_normal and value !=0:
                print(f"{column} a une valeur différente de la valeur moyenne enregistrée de {value}%")
                check_list[column] =  False
              else:
                  
                  check_list[column] =  None
        elif column in ['Biotine(µg)', 'Fluorure(mg)', 'Chrome(µg)', 'Molybdene(µg)']:
            print(f"{column} pas testé")
            check_list[column] =  None
    return  check_list

def value_only_numerical(value):
    value = value.replace(',', '.')
    characters = ".1234567890"
    return value == "".join(x for x in value if x in characters) and value != ""

def check_total_weight(df_pred, max_total_weight = 100):
    weight = 0
    for column in COLUMNS_FOR_TOTAL_WEIGHT:
        value = df_pred[column].to_list()[0]
        if value_only_numerical(value):
          weight += float(value)
    return  weight <= max_total_weight

def check_fat_weight(df_pred):
    fat = COLUMNS_FOR_FAT_WEIGHT[0]
    fat_value = df_pred[fat].to_list()[0]
    if value_only_numerical(fat_value):
        fat_value = float(fat_value)
    else:
        fat_value = 0
    sum_fat_weight = 0
    for column in COLUMNS_FOR_FAT_WEIGHT[1:]:
        value = df_pred[column].to_list()[0]
        if value_only_numerical(value):
          sum_fat_weight += float(value)
    return  sum_fat_weight <= fat_value

def check_carbohydrate_weight(df_pred):
    carb = COLUMNS_FOR_CARBOHYDRATE_WEIGHT[0]
    carb_value = df_pred[carb].to_list()[0]
    if value_only_numerical(carb_value):
        carb_value = float(carb_value)
    else:
        carb_value = 0
    sum_carb_weight = 0
    for column in COLUMNS_FOR_CARBOHYDRATE_WEIGHT[1:]:
        value = df_pred[column].to_list()[0]
        if value_only_numerical(value):
          sum_carb_weight += float(value)
    return  sum_carb_weight <= carb_value

def pred_energie( mat_grasse, glucides, polyols, fibre, prot  ):
    pred_kj = 17 * (glucides - polyols) + 10*polyols + 17*prot + 37*mat_grasse + 8*fibre
    pred_kcal = 4 * (glucides - polyols) + 2.4*polyols + 4*prot + 9*mat_grasse + 2*fibre
    return pred_kj,pred_kcal

def check_energy(df_pred, esp = 0.1):
    estimates = df_pred[COLUMNS_FOR_ENERGY].loc[0].to_list()
    estimates_float = []
    for estimate in estimates:
        if value_only_numerical(estimate):
            estimates_float.append(float(estimate))
        else:
            estimates_float.append(0)
    pred_kj, pred_kcal = pred_energie(estimates_float[2],estimates_float[3], estimates_float[4],estimates_float[5], estimates_float[6] )
    gap_kj = relative_gap(pred_kj, estimates_float[0])
    gap_kcal = relative_gap(pred_kcal, estimates_float[1])

    return (gap_kj < esp, gap_kcal < esp)


def logical_check(df_pred, esp):
    
    doublets = []
    bool_total_weigth = check_total_weight(df_pred, max_total_weight = 100)

    doublets.append((COLUMNS_FOR_TOTAL_WEIGHT, bool_total_weigth))
    
    bool_weight_fat = check_fat_weight(df_pred)

    doublets.append((COLUMNS_FOR_FAT_WEIGHT, bool_weight_fat))

    bool_carbo_weight = check_carbohydrate_weight(df_pred)

    doublets.append((COLUMNS_FOR_CARBOHYDRATE_WEIGHT, bool_carbo_weight))

    bool_energy_kj, bool_energy_kcal = check_energy(df_pred, esp)

    doublets.append(([COLUMNS_FOR_ENERGY[0]] + COLUMNS_FOR_ENERGY[2:] , bool_energy_kj))
    doublets.append((COLUMNS_FOR_ENERGY[1:] , bool_energy_kcal))


    dic = {column : True for column in TOTAL_COLUMNS}

    for doublet in doublets:
        for column in doublet[0]:
            dic[column] = dic[column] and doublet[1]
    
    return dic

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file from path {file_path} : {e}")
        return None
    
def check(df_pred,code_product, eps_energy = 0.05, eps_ciqual = 0.5 ):

    other_check_dic = logical_check(df_pred, eps_energy)

    df = read_excel(PATH_CIQUAL_TABLE)
    ciqual_check_dic = check_with_ciqual(df_pred, df, code_product,eps_ciqual, CONVERTER)
    
    for key in other_check_dic.keys():
        ciqual_check_dic[key] = other_check_dic[key] and ciqual_check_dic[key]
    return ciqual_check_dic

def check_from_config(df_pred, config_path):
    config = read_config_predict(config_path)
    config.get("file_data")
    return check(df_pred, 
          code_product= config.get("code_product"), 
          eps_energy = config.get("eps_energy"), 
          eps_ciqual = config.get("eps_ciqual") )

def create_result_table(nom_estimates, value_estimates, check_estimate, config_path):
    config = read_config_predict(config_path)
    df = pd.DataFrame({
        'Nom': nom_estimates,
        'Valeur': value_estimates.values(),
        'Check': check_estimate.values()
    })
    
    path_save = config.get("path_save")
    if path_save != "" and os.path.exists(os.path.dirname(path_save)):
            df.to_csv(path_save)
    elif not os.path.exists(os.path.dirname(path_save)):
        print("Invalid or inaccessible path_save")
    return df
    

