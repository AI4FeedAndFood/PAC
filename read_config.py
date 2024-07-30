import os
import json 

def read_config_predict(path_config):
    """
    Reads and validates a JSON configuration file for prediction settings.

    This function loads a JSON file, extracts specific configuration parameters,
    and performs type checking on these parameters. It's designed to ensure that
    the configuration file contains all necessary information in the correct format
    for prediction tasks.

    Args:
    path_config (str): The file path to the JSON configuration file.

    Returns:
    dict: A dictionary containing the validated configuration parameters.

    Raises:
    FileNotFoundError: If the specified configuration file does not exist.
    ValueError: If any of the configuration parameters are missing or have incorrect types.
    Exception: For any other errors encountered during the process.

    The function checks for the following configuration parameters:
    - model (str): The model name or path.
    - model_yolo (str): The YOLO model name or path.
    - file_data (str): Path to the data file.
    - code_product_eurofins (str or False): Eurofins product code.
    - code_product_ciqual (int or False): CIQUAL product code.
    - max_seq_length (int): Maximum sequence length for the model.
    - dtype (str): Data type for model computations.
    - load_in_4bit (bool): Whether to load the model in 4-bit precision.
    - path_save (str): Path to save the results.
    - coef_deviation_eurofins (float): Coefficient value for Eurofins calculations.
    - eps_energy (float): Epsilon value for energy calculations.
    - eps_ciqual (float): Epsilon value for CIQUAL calculations.

    Note:
    The function will print an error message and return the config dictionary
    even if an exception is raised during the process.
    """
    try: 
        if not os.path.isfile(path_config):
            raise FileNotFoundError(f"The config file '{path_config}' does not exist.")

        with open(path_config, 'r') as f:
            config = json.load(f)

        model = config.get("model")
        model_yolo = config.get("model_yolo")
        file_data = config.get("file_data")
        code_product_eurofins = config.get("code_product_eurofins")
        code_product_ciqual = config.get("code_product_ciqual")
        
        max_seq_length = config.get("max_seq_length")
        dtype = config.get("dtype")
        load_in_4bit = config.get("load_in_4bit")
        path_save = config.get("path_save")
        coef_deviation_eurofins = config.get("coef_deviation_eurofins")
        eps_energy = config.get("eps_energy")
        eps_ciqual = config.get("eps_ciqual")
    

        # Add type and value checks for the loaded data
        if not isinstance(model, str):
            raise ValueError("The 'model' in the config file must be a string.")
        if not isinstance(model_yolo, str):
            raise ValueError("The 'model_yolo' in the config file must be a string.")

        if not isinstance(file_data, str):
            raise ValueError("The 'file_data' in the config file must be a string.")
        
        if not isinstance(code_product_ciqual, int) or code_product_ciqual == False:
            raise ValueError("The 'code_product_ciqual' in the config file must be a integer or false.")
        
        if not isinstance(code_product_eurofins, str) and code_product_eurofins != False:
            raise ValueError("The 'code_product_eurofins' in the config file must be a str or false.")

        if not isinstance(max_seq_length, int):
            raise ValueError("The 'max_seq_length' in the config file must be an integer.")

        if not isinstance(dtype, str):
            raise ValueError("The 'dtype' in the config file must be a string.")

        if not isinstance(load_in_4bit, bool):
            raise ValueError("The 'load_in_4bit' in the config file must be a boolean.")

        if not isinstance(path_save, str):
            raise ValueError("The 'path_save' in the config file must be a string.")

        if not isinstance(eps_energy, float):
            raise ValueError("The 'eps_energy' in the config file must be a float.")

        if not isinstance(eps_ciqual, float):
            raise ValueError("The 'eps_ciqual' in the config file must be a float.")
    
        if not isinstance(coef_deviation_eurofins, float) and not isinstance(coef_deviation_eurofins, int):
            raise ValueError("The 'coef_deviation_eurofins' in the config file must be a float or a int.")
        
        return config
    
    except Exception as e:
        print(f"Error in load_config: {e}")
        return config