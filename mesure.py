"""
This module contains functions for computing accuracy metrics and testing data predictions against target values.

Functions:
- compute_acc1: Computes the first accuracy metric (ACC1)
- compute_acc2: Computes the second accuracy metric (ACC2)
- compute_acc3: Computes the third accuracy metric (ACC3)
- test_datas: Tests predictions against target values and computes accuracy metrics
- test_csv: Reads CSV files and tests predictions against target values

Dependencies:
- math
- pandas
- constant (COLUMNS)

"""
import math 
import pandas as pd 
from constant import COLUMNS 

def compute_acc1(y_pred, y_target, for_standard_parameters = False):
    """
    Computes the first accuracy metric (ACC1).

    This function checks if prediction and target values are both empty or both non-empty.

    Args:
    y_pred (list): Predicted values
    y_target (list): Target values
    for_standard_parameters (bool): If True, only considers the first 13 parameters

    Returns:
    float: ACC1 score
    """
    n_min = min(len(y_pred), len(y_target))
    
    if for_standard_parameters:
        n_min = min(n_min, 13) #there are 13 standard parameters Energ(kJ)x2, Fatx4, Sugarx4, Prot, Fiber and Salt 
    k = 0
    for l in range(n_min):
        target_value = y_target[l]
        pred_value = y_pred[l]
        if isinstance(pred_value, str) and isinstance(pred_value, str):
            if (pred_value == '' and target_value == ''): #check if both empty
                k += 1
            elif (pred_value != '' and target_value != ''): #check if both full
                k += 1
        elif (isinstance(pred_value, int) or isinstance(pred_value, float)) and (isinstance(target_value, int) or isinstance(target_value, float)):
            if math.isnan(pred_value) and math.isnan(target_value):
                k += 1  #check if both empty
            elif not math.isnan(pred_value) and not math.isnan(target_value):
                k += 1 #check if both full
        
        elif isinstance(pred_value, int) or isinstance(pred_value, float):
            if math.isnan(pred_value) and target_value == "":
                k += 1  #check if both empty
            elif not math.isnan(pred_value) and target_value != "":
                k += 1 #check if both full
        elif isinstance(target_value, int) or isinstance(target_value, float):
            if math.isnan(target_value) and pred_value == "":
                k += 1  #check if both empty
            if not math.isnan(target_value) and pred_value != "":
                k += 1  #check if both empty
        
    return k/n_min

def compute_acc2(y_pred, y_target, for_standard_parameters = False):
    """
    Computes the second accuracy metric (ACC2).

    This function checks if non-empty prediction values match the target values.

    Args:
    y_pred (list): Predicted values
    y_target (list): Target values
    for_standard_parameters (bool): If True, only considers the first 13 parameters

    Returns:
    float: ACC2 score
    """
    n_min = min(len(y_pred), len(y_target))
    if for_standard_parameters:
        n_min = min(n_min, 13) #there are 13 standard parameters Energ(kJ)x2, Fatx4, Sugarx4, Prot, Fiber and Salt 
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
                elif str(y_pred[l]).strip().lower() in "trace" and   str(y_target[l]).strip().lower() in "trace":
                    k+=1
                elif len(str(y_pred[l]))>0 and len(str(y_target[l]))>0 and str(y_pred[l])[0] == '<' and  str(y_target[l][0]) == '<':
                    if float(y_pred[l][1:]) == float(y_target[l][1:]):
                        k+=1
    if n_max != 0:
        return k/n_max
    else:
        return 1
    
def compute_acc3(acc1, acc2):
    """
    Computes the third accuracy metric (ACC3).

    Returns 1 if both ACC1 and ACC2 are 1, otherwise returns 0.

    Args:
    acc1 (float): ACC1 score
    acc2 (float): ACC2 score

    Returns:
    int: ACC3 score (0 or 1)
    """
    if acc1 == 1 and acc2 == 1:
        return 1 
    else: 
        return 0 



def test_datas(y_preds, y_targets):
    """
    Tests predictions against target values and computes accuracy metrics.

    Args:
    y_preds (list): List of prediction lists
    y_targets (list): List of target lists

    Returns:
    dict: Dictionary containing various accuracy metrics
    """
    dic_accuracy = {}
    acc1_tot = 0
    acc2_tot = 0
    acc3_tot = 0
    
    acc1_std_tot = 0 
    acc2_std_tot = 0 
    acc3_std_tot = 0 
    
    n = len(y_preds)
    if len(y_preds) != len(y_targets):
        print(f"Warning for function test_datas in mesure.py: {len(y_preds)=} and {len(y_targets)=} are different")
    
    for y_pred, y_target in zip(y_preds, y_targets):
        acc1 = compute_acc1(y_pred, y_target)
        acc2 = compute_acc2(y_pred, y_target)
        acc3 = compute_acc3(acc1, acc2)
        
        acc1_standard = compute_acc1(y_pred, y_target, for_standard_parameters= True)
        acc2_standard = compute_acc2(y_pred, y_target, for_standard_parameters= True)
        acc3_standard = compute_acc3(acc1_standard, acc2_standard)

        acc1_tot += acc1/n
        acc2_tot += acc2/n
        acc3_tot += acc3/n
        
        acc1_std_tot += acc1_standard/n
        acc2_std_tot += acc2_standard/n
        acc3_std_tot += acc3_standard/n
        
        

    dic_accuracy['acc1_tot'] = acc1_tot
    dic_accuracy['acc2_tot'] = acc2_tot
    dic_accuracy['acc3_tot'] = acc3_tot
    dic_accuracy['acc1_std_tot'] = acc1_std_tot
    dic_accuracy['acc2_std_tot'] = acc2_std_tot
    dic_accuracy['acc3_std_tot'] = acc3_std_tot
    
    return dic_accuracy

def test_csv(csv_pred, csv_target):
    """
    Reads CSV files (or pandas DataFrames) and tests predictions against target values.

    Args:
    csv_pred (str or pd.DataFrame): Path to prediction CSV file or pandas DataFrame
    csv_target (str or pd.DataFrame): Path to target CSV file or pandas DataFrame

    Returns:
    dict: Dictionary containing various accuracy metrics
    """
    if isinstance(csv_pred, pd.DataFrame):
        df_pred = csv_pred 
    else:
        df_pred = pd.read_csv(csv_pred)
        
    if isinstance(csv_target, pd.DataFrame):
        df_target = csv_target 
    else:
        df_target = pd.read_csv(csv_target)

    y_preds = []
    y_targets = []
    for row1, row2 in zip(df_pred.iterrows(), df_target.iterrows()):
        y_pred = list(row1[1][COLUMNS])
        y_target = list(row2[1][COLUMNS])
        y_preds.append(y_pred)
        y_targets.append(y_target)
        
    return test_datas(y_preds, y_targets)
        
    


