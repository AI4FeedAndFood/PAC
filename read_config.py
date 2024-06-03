import os
import json 

def read_config_predict(path_config):
    try: 
        if not os.path.isfile(path_config):
            raise FileNotFoundError(f"The config file '{path_config}' does not exist.")

        with open(path_config, 'r') as f:
            config = json.load(f)

        model = config.get("model")
        file_data = config.get("file_data")
        code_product = config.get("code_product")
        max_seq_length = config.get("max_seq_length")
        dtype = config.get("dtype")
        load_in_4bit = config.get("load_in_4bit")
        path_save = config.get("path_save")
        eps_energy = config.get("eps_energy")
        eps_ciqual = config.get("eps_ciqual")

        # Add type and value checks for the loaded data
        if not isinstance(model, str):
            raise ValueError("The 'model' in the config file must be a string.")

        if not isinstance(file_data, str):
            raise ValueError("The 'file_data' in the config file must be a string.")

        if not isinstance(code_product, int):
            raise ValueError("The 'code_product' in the config file must be a integer.")

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
        print(config)
        return config
    except Exception as e:
        print(f"Error in load_config: {e}")