import pandas as pd 
from constant import PATH_CIQUAL_TABLE,PATH_EUROFINS_TABLE, COLUMN_CONVERTER_CIQUAL,COLUMNS_FOR_ENERGY,COLUMNS_FOR_CARBOHYDRATE_WEIGHT, COLUMNS_FOR_FAT_WEIGHT, COLUMNS_FOR_TOTAL_WEIGHT, COLUMNS, COLUMN_CONVERTER_EUROFINS  
import os 
from read_config import read_config_predict
import warnings
    
def convert_to_float(value):
    """
    Converts a value to a float. Handles different data types and formats.
    
    Parameters:
    value (int, float, str): The value to be converted.
    
    Returns:
    float: The converted float value.
    
    Raises:
    TypeError: If the value is not an int, float, or str.
    """
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    elif isinstance(value, str):
        value = value.replace(',', '.')
        characters = ".1234567890"
        try:
            if 'trace' in value.strip().lower():
                return 0
            elif '-' in value or value == 'nan' or value == '':
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
    else:
        raise TypeError("value must be a int or a float a string")
      
def relative_gap(y_pred, y_target):
    """
    Calculates the relative gap between predicted and target values.
    
    Parameters:
    y_pred (float): The predicted value.
    y_target (float): The target value.
    
    Returns:
    float: The relative gap.
    """
    return abs(y_pred - y_target)/(y_target + 0.001)

def is_a_normal_value_for_nut_name(y_pred, y_target, max_relative_gab):
    """
    Checks if a predicted value for a nutrient is within an acceptable range of the target value.
    
    Parameters:
    y_pred (float, str): The predicted value.
    y_target (float, str): The target value.
    max_relative_gap (float): The maximum acceptable relative gap.
    
    Returns:
    tuple: (bool indicating if the value is normal, the calculated relative gap)
    """
    if y_target == -1:
        return False, 0
    else:
        if isinstance(y_target, str):
            y_target = convert_to_float(y_target)
        if isinstance(y_pred, str):
            y_pred = convert_to_float(y_pred)
    if y_target >= 0 and y_pred >=  0:
        eps = relative_gap(y_pred, y_target)
        if eps < max_relative_gab or abs(y_target - y_pred) <= 0.5:
            return True, eps
        else:
            return False, eps
    else:
      return False, 0
  
def extract_from_ciqual(df, alim_code_ciqual, nut_name_ciqual):
    """
    Extracts a nutrient value from the CIQUAL dataset.
    
    Parameters:
    df (pd.DataFrame): The CIQUAL dataset.
    alim_code_ciqual (int): The aliment code in the CIQUAL dataset.
    nut_name_ciqual (str): The nutrient name in the CIQUAL dataset.
    
    Returns:
    float: The extracted nutrient value or -1 if not found.
    
    Raises:
    ValueError: If df is not a pandas DataFrame.
    TypeError: If alim_code_ciqual is not an int or nut_name_ciqual is not a string.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(alim_code_ciqual, int):
        raise TypeError("alim_code must be a int")
    if not isinstance(nut_name_ciqual, str):
        raise TypeError("nut_name must be a string")
    
    if alim_code_ciqual not in df['alim_code'].to_list():
        warnings.warn("alim_code not in the Ciqual table")
    if nut_name_ciqual not in df.columns:
        warnings.warn("nut_name not in the Ciqual columns, maibe updtate converter from constant or check Ciqual Table")
    try:
        valeur = df.loc[df['alim_code'] == alim_code_ciqual, nut_name_ciqual].values[0]
        return valeur
    except IndexError:
        warnings.warn(f"Aucune valeur trouvée pour cet alim_code {alim_code_ciqual} et nut_name {nut_name_ciqual}.")
        return -1
    
def check_with_ciqual(dict_pred, df_true,alim_code, max_relative_gab, converter):
    """
    Checks the predicted values against the CIQUAL dataset.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    df_true (pd.DataFrame): The CIQUAL dataset.
    alim_code (int): The aliment code in the CIQUAL dataset.
    max_relative_gap (float): The maximum acceptable relative gap.
    converter (dict): The converter dictionary for nutrient names.
    
    Returns:
    dict: A dictionary indicating the consistency of each nutrient.
    """
    check_dict = {}
    for column in COLUMNS:

        if column in converter.keys():
              nut_name_ciqual = converter[column] #Nut_name in ciqual 
              y_target = extract_from_ciqual(df_true, alim_code, nut_name_ciqual)
              is_normal, value = is_a_normal_value_for_nut_name(dict_pred[column], y_target, max_relative_gab)
              value = round(value*100) #convertion en % avec arrondi
              if is_normal:
                  print(f"{column} a une valeur attendue cohérente")
                  check_dict[column] =  True
              elif not is_normal and value !=0:
                print(f"{column} a une valeur différente de la valeur moyenne enregistrée de {value}%")
                check_dict[column] =  False
              else:
                  
                  check_dict[column] =  None
        else:
            print(f"{column} not tested.") #Because not in converter ciqual
            check_dict[column] =  None
    return  check_dict

def extract_from_eurofins(df, product_code, parameter_name):
    """
    Extracts a parameter value from the Eurofins dataset.
    
    Parameters:
    df (pd.DataFrame): The Eurofins dataset.
    product_code (str): The product code in the Eurofins dataset.
    parameter_name (str): The parameter name in the Eurofins dataset.
    
    Returns:
    tuple: (average value, standard deviation) or (-1, -1) if not found.
    
    Raises:
    ValueError: If df is not a pandas DataFrame.
    TypeError: If product_code is not a string or parameter_name is not a string.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(product_code, str):
        raise TypeError("alim_code must be a str")
    if not isinstance(parameter_name, str):
        raise TypeError("nut_name must be a string")
    
    if product_code not in df['ProductCode'].to_list():
        warnings.warn("ProductCode not in the Eurofins table")
        return -1, -1
    if parameter_name not in df['ParameterName'].to_list():
        warnings.warn(f"Parameter name : {parameter_name} not in the Eurofins columntables, maibe updtate converter from constant or check Eurofins Table")
        return -1, -1
    try:
        filtered = df[(df['ProductCode'] == product_code) & (df['ParameterName'] == parameter_name)]
        if filtered.empty:
            print( f"Aucune valeur trouvée pour cet product_code {product_code} et parameter_name {parameter_name}.")
            return -1, -1
        
        average = filtered['average'].iloc[0]
        std_dev = filtered['std_deviation'].iloc[0]
    
        return average, std_dev
    except IndexError:
        warnings.warn( f"Aucune valeur trouvée pour cet product_code {product_code} et parameter_name {parameter_name}.")
        return -1, -1
    
def check_with_eurofins(dict_pred, df_true, product_code, coef_deviation, converter):
    """
    Checks the predicted values against the Eurofins dataset.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    df_true (pd.DataFrame): The Eurofins dataset.
    product_code (str): The product code in the Eurofins dataset.
    coef_deviation (float): The coefficient for acceptable deviation.
    converter (dict): The converter dictionary for parameter names.
    
    Returns:
    dict: A dictionary indicating the consistency of each parameter.
    """
    check_dict = {}
    for column in COLUMNS:

        if column in converter.keys() and dict_pred[column] != "":
            product_name = converter[column]
            average, deviation = extract_from_eurofins(df_true, product_code, product_name)
            if average == -1 or deviation == -1: #no values in the datas
                check_dict[column] =  None
                 #print(f"{column} not tested.") #Because not in converter eurofins
            else:
                y_pred = convert_to_float(dict_pred[column])
                average = convert_to_float(average)
                deviation = convert_to_float(deviation)
                
                if y_pred > (average - deviation*coef_deviation) and y_pred < (average + deviation * coef_deviation):
                    check_dict[column] =  True
                else: 
                    print(f"For {column}, is value {y_pred} not in the averages values {[(average - deviation*coef_deviation), (average + deviation * coef_deviation)]}.")
                    check_dict[column] =  False
        elif column in converter.keys() and dict_pred[column] == "":
            check_dict[column] =  None
        else:
            check_dict[column] =  None
            #print(f"{column} not tested.") #Because not in converter eurofins
                            
    return  check_dict

def value_only_numerical(value):
    """
    Checks if the value is purely numerical (int or float).
    
    Parameters:
    value (any): The value to check.
    
    Returns:
    bool: True if the value is numerical, False otherwise.
    """
    if  isinstance(value, float) or isinstance(value, int):
        return True
    else:
        value = value.replace(',', '.')
        characters = ".1234567890"
        return value == "".join(x for x in value if x in characters) and value != ""

def check_total_weight(dict_pred, max_total_weight = 100):
    """
    Checks if the total weight of the components is within the maximum allowed weight.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    max_total_weight (float): The maximum allowed total weight.
    
    Returns:
    bool: True if the total weight is within the limit, False otherwise.
    """
    weight = 0
    for column in COLUMNS_FOR_TOTAL_WEIGHT:
        value = dict_pred[column]
        if value_only_numerical(value):
          weight += float(value)
    return  weight <= max_total_weight

def check_fat_weight(dict_pred):
    """
    Checks if the sum of fat components is within the total fat value.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    
    Returns:
    bool: True if the sum of fat components is within the total fat value, False otherwise.
    """
    fat = COLUMNS_FOR_FAT_WEIGHT[0]
    fat_value = dict_pred[fat]
    if value_only_numerical(fat_value):
        fat_value = float(fat_value)
    else:
        fat_value = 0
    sum_fat_weight = 0
    for column in COLUMNS_FOR_FAT_WEIGHT[1:]:
        value = dict_pred[column]
        if value_only_numerical(value):
          sum_fat_weight += float(value)
    return  sum_fat_weight <= fat_value

def check_carbohydrate_weight(dict_pred):
    """
    Checks if the sum of carbohydrate components is within the total carbohydrate value.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    
    Returns:
    bool: True if the sum of carbohydrate components is within the total carbohydrate value, False otherwise.
    """
    carb = COLUMNS_FOR_CARBOHYDRATE_WEIGHT[0]
    carb_value = dict_pred[carb]
    if value_only_numerical(carb_value):
        carb_value = float(carb_value)
    else:
        carb_value = 0
    sum_carb_weight = 0
    for column in COLUMNS_FOR_CARBOHYDRATE_WEIGHT[1:]:
        value = dict_pred[column]
        if value_only_numerical(value):
          sum_carb_weight += float(value)
    return  sum_carb_weight <= carb_value

def pred_energie( mat_grasse, glucides, polyols, fibre, prot  ):
    """
    Predicts the energy in kJ and kcal based on macronutrient values. According to European documentations.
    
    Parameters:
    mat_grasse (float): The fat content.
    glucides (float): The carbohydrate content.
    polyols (float): The polyol content.
    fibre (float): The fiber content.
    prot (float): The protein content.
    
    Returns:
    tuple: (predicted energy in kJ, predicted energy in kcal)
    """
    pred_kj = 17 * (glucides - polyols) + 10*polyols + 17*prot + 37*mat_grasse + 8*fibre
    pred_kcal = 4 * (glucides - polyols) + 2.4*polyols + 4*prot + 9*mat_grasse + 2*fibre
    return pred_kj,pred_kcal

def check_energy(dict_pred, esp = 0.1):
    """
    Checks if the predicted energy values are within an acceptable range.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    esp (float): The acceptable relative gap.
    
    Returns:
    tuple: (bool indicating if kJ value is acceptable, bool indicating if kcal value is acceptable)
    """
    estimates_float = []
    for column in COLUMNS_FOR_ENERGY:
        if value_only_numerical(dict_pred[column]):
            estimates_float.append(float(dict_pred[column]))
        else:
            estimates_float.append(0)
    pred_kj, pred_kcal = pred_energie(estimates_float[2],estimates_float[3], estimates_float[4],estimates_float[5], estimates_float[6] )
    gap_kj = relative_gap(pred_kj, estimates_float[0])
    gap_kcal = relative_gap(pred_kcal, estimates_float[1])

    return (gap_kj < esp, gap_kcal < esp)


def logical_check(dict_pred, esp):
    """
    Performs logical consistency checks on the predicted values.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    esp (float): The acceptable relative gap for energy checks.
    
    Returns:
    dict: A dictionary indicating the logical consistency of each component.
    """    
    doublets = []
    bool_total_weigth = check_total_weight(dict_pred, max_total_weight = 100)

    doublets.append((COLUMNS_FOR_TOTAL_WEIGHT, bool_total_weigth))
    
    bool_weight_fat = check_fat_weight(dict_pred)

    doublets.append((COLUMNS_FOR_FAT_WEIGHT, bool_weight_fat))

    bool_carbo_weight = check_carbohydrate_weight(dict_pred)

    doublets.append((COLUMNS_FOR_CARBOHYDRATE_WEIGHT, bool_carbo_weight))

    bool_energy_kj, bool_energy_kcal = check_energy(dict_pred, esp)

    doublets.append(([COLUMNS_FOR_ENERGY[0]] + COLUMNS_FOR_ENERGY[2:] , bool_energy_kj))
    doublets.append((COLUMNS_FOR_ENERGY[1:] , bool_energy_kcal))


    dic = {column : None for column in COLUMNS}

    for doublet in doublets:
        for column in doublet[0]:
            if dic[column] == None:
                dic[column] = doublet[1]
            else:
                dic[column] = dic[column] and doublet[1]
    
    return dic

def read_excel(file_path):
    """
    Reads an Excel file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the Excel file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the data from the Excel file.
    
    Raises:
    Exception: If there is an error reading the file.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file from path {file_path} : {e}")
        return None
    
def check(dict_pred,code_product_eurofins = False, code_product_ciqual = False, eps_energy = 0.05, eps_ciqual = 0.5, coef_deviation_eurofins = 0.5, loaded_df_eurofins = False,loaded_df_ciqual = False  ):
    """
    Performs all checks (logical, CIQUAL, and Eurofins) on the predicted values.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    code_product_eurofins (bool or str): The product code for Eurofins data, or False if not available.
    code_product_ciqual (bool or int): The aliment code for CIQUAL data, or False if not available.
    eps_energy (float): The acceptable relative gap for energy checks.
    eps_ciqual (float): The acceptable relative gap for CIQUAL checks.
    coef_deviation_eurofins (float): The coefficient for acceptable deviation in Eurofins checks.
    loaded_df_eurofins (bool or pd.DataFrame): The preloaded Eurofins DataFrame, or False if not available.
    loaded_df_ciqual (bool or pd.DataFrame): The preloaded CIQUAL DataFrame, or False if not available.
    
    Returns:
    dict: A dictionary containing the results of all checks.
    """
    dicts_for_check = {
        'logical' : {},
        'ciqual' : {},
        'eurofins' : {}
    }
    
    logical_check_dict = logical_check(dict_pred, eps_energy)
    dicts_for_check['logical'] = logical_check_dict

      
    if code_product_ciqual != False:
        
        if isinstance(loaded_df_ciqual, pd.DataFrame):
            df_ciqual = loaded_df_ciqual 
        elif loaded_df_ciqual == False:
            df_ciqual = read_excel(PATH_CIQUAL_TABLE)
        else:
            print(f"Function check get 'loaded_df_ciqual', but it's not a pd.DataFrame, so Ciqual will be loaded using the path located in constant.py")
            df_ciqual = read_excel(PATH_CIQUAL_TABLE)
            
        ciqual_check_dict = check_with_ciqual(dict_pred, df_ciqual, code_product_ciqual,eps_ciqual, COLUMN_CONVERTER_CIQUAL)
        dicts_for_check['ciqual'] = ciqual_check_dict
    
    if code_product_eurofins != False:

        if isinstance(loaded_df_eurofins, pd.DataFrame):
            df_eurofins = loaded_df_eurofins
        elif loaded_df_eurofins == False:
            df_eurofins = read_excel(PATH_EUROFINS_TABLE)
        else:
            print(f"Function check get 'loaded_df_eurofins', but it's not a pd.DataFrame, so Eurofins Table will be loaded using the path located in constant.py")
            df_eurofins = read_excel(PATH_EUROFINS_TABLE)

        eurofins_check_dict = check_with_eurofins(dict_pred, df_eurofins, code_product_eurofins,coef_deviation_eurofins, COLUMN_CONVERTER_EUROFINS)
        dicts_for_check['eurofins'] = eurofins_check_dict

    

    return dicts_for_check

def check_from_config(dict_pred, config_path):
    """
    Performs checks based on configuration file settings.
    
    Parameters:
    dict_pred (dict): The dictionary of predicted values.
    config_path (str): The path to the configuration file.
    
    Returns:
    dict: A dictionary containing the results of all checks.
    """
    config = read_config_predict(config_path)
    config.get("file_data")
    return check(dict_pred, 
          code_product_eurofins= config.get("code_product_eurofins"), 
          code_product_ciqual= config.get("code_product_ciqual"), 
          eps_energy = config.get("eps_energy"), 
          eps_ciqual = config.get("eps_ciqual"),
          coef_deviation_eurofins = config.get("coef_deviation_eurofins"))

def dict_to_values(dict):
    """
    Converts a dictionary's values to a list, ensuring the dictionary's length matches a predefined length.

    Parameters:
    dict (dict): The input dictionary to extract values from.

    Returns:
    list or None: A list of the dictionary's values if the dictionary is not empty and its length matches
    the predefined length. Otherwise, returns None.

    Notes:
    - The function checks if the length of the input dictionary matches the length of a predefined constant `COLUMNS`.
    - If the length of the dictionary does not match the length of `COLUMNS`, a warning is issued.
    - If the dictionary is empty, the function returns None.
    """
    n = len(COLUMNS)
    
    if len(dict) > 0:
        values = dict.values()
        if len(dict) != n:
            warnings.warn(f"lenght of dicts_check_estimate  {len(dict)} != lenght of COLUMNS  {n}")    
    else: 
        values = None
    return values
    
def create_result_table(nom_estimates, value_estimates, dicts_check_estimate, config_path):
    """
    Creates a result table from the estimates and check results.
    
    Parameters:
    nom_estimates (list): The list of estimate names.
    value_estimates (dict): The dictionary of estimate values.
    dicts_check_estimate (dict): The dictionary containing check results.
    config_path (str): The path to the configuration file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the result table.
    """
    config = read_config_predict(config_path)    

    values_check_eurofins = dict_to_values(dicts_check_estimate['eurofins'])
    values_check_ciqual = dict_to_values(dicts_check_estimate['ciqual'])
    values_check_logical = dict_to_values(dicts_check_estimate['logical'])
    
    df = pd.DataFrame({
        'Nom': nom_estimates,
        'Valeur': value_estimates.values(),
        'Eurofins Consistency': values_check_eurofins,
        'Ciqual Consistency':  values_check_ciqual,
        'Logical Consistency': values_check_logical
    })
    
    path_save = config.get("path_save")
    if path_save != "" and os.path.exists(os.path.dirname(path_save)):
            df.to_csv(path_save)
    elif not os.path.exists(os.path.dirname(path_save)):
        print("Invalid or inaccessible path_save")
    return df
    

