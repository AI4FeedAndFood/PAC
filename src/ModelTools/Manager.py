import os
import joblib
import torch
import json
import torch.nn as nn

from Models import  MLP, SingleFC, FCEncoder, CustomBertEncoder, ImageEncodeur, MultiModalClassifier

class ModelManager:
    def __init__(self, training_name, base_path="ModelWeights"):
        self.base_path = base_path
        self.training_name = training_name
        self.full_path = os.path.join(self.base_path, self.training_name)

        # Creat the file
        os.makedirs(self.full_path, exist_ok=True)

    def save_encoder(self, encoder, encoder_name):
        joblib_file = os.path.join(self.full_path, encoder_name + '.pkl')
        joblib.dump(encoder, joblib_file)
        print(f"Encoder '{encoder_name}' saved in {joblib_file}")

    def load_ohencoder(self, encoder_name):
        encoder_path = os.path.join(self.full_path, encoder_name + '.pkl')
        encoder = joblib.load(encoder_path)
        print(f"Encoder '{encoder_name}' loaded from {encoder_path}")
        return encoder

    def save_training_params(self, params, training_name):
        params_file = os.path.join(self.full_path, f"{training_name}_params.json")
        with open(params_file, 'w') as f:
            f.write(str(params))
        print(f"Training parameters saved in {params_file}")

    def get_onehot_categories(self, encoder):
        if not hasattr(encoder, 'categories_'):
            raise AttributeError("The encoder doesn't seem to be a OneHotEncoder or hasn't been fitted to data yet.")
        categories = encoder.categories_
        if len(categories) == 1:
            return categories[0].tolist()
        return [cat.tolist() for cat in categories]
    
    def save_model_state_dict(self, model, model_name):
        state_dict_file = os.path.join(self.full_path, f"{model_name}_state_dict.pth")
        torch.save(model.state_dict(), state_dict_file)
        print(f"Model state_dict for '{model_name}' saved in {state_dict_file}")
    
        # Save model configuration
        model_config = {
            "encoders": {
                name: self._get_encoder_config(encoder)
                for name, encoder in model.encoders.items()
            },
            "head": self._get_head_config(model.head)
        }

        config_file = os.path.join(self.full_path, f"{model_name}_config.json")
        with open(config_file, "w") as f:
            json.dump(model_config, f)
        print(f"Model configuration for '{model_name}' saved in {config_file}")

    def _get_encoder_config(self, encoder):
        if isinstance(encoder, FCEncoder):
            return {
                "type": "FCEncoder",
                "onehot_dim": encoder.embedding.in_features,
                "embedding_dim": encoder.embedding.out_features
            }
        elif isinstance(encoder, CustomBertEncoder):
            return {
                "type": "CustomBertEncoder",
                "bert_model_name": encoder.CustomBertEncoder.config._name_or_path,
                "n_attention_layers": len(encoder.CustomBertEncoder.encoder.layer)
            }
        elif isinstance(encoder, ImageEncodeur):
            return {
                "type": "ImageEncodeur",
                "weights": "IMAGENET1K_V1",  # Assuming default weights
                "output_dim": encoder.imageEncodeur.fc.out_features
            }
        elif isinstance(encoder, nn.Identity):
            return {"type": "Identity"}
        else:
            raise ValueError(f"Unknown encoder type: {type(encoder)}")

    def _get_head_config(self, head):
        if isinstance(head, MLP):
            return {
                "type": "MLP",
                "input_dim": head.bn.num_features,
                "num_classes": head.classifier[-1].out_features
            }
        elif isinstance(head, SingleFC):
            return {
                "type": "SingleFC",
                "input_dim": head.classifier.in_features,
                "num_classes": head.classifier.out_features
            }
        else:
            raise ValueError(f"Unknown head type: {type(head)}")

    def load_model(self, model_name):
        config_file = os.path.join(self.full_path, f"{model_name}_config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
    
        encoders = nn.ModuleDict({
            name: self._create_encoder(encoder_config)
            for name, encoder_config in config["encoders"].items()
        })
    
        head = self._create_head(config["head"])
    
        model = MultiModalClassifier(encoders=encoders, head=head)
    
        state_dict_file = os.path.join(self.full_path, f"{model_name}_state_dict.pth")
        model.load_state_dict(torch.load(state_dict_file))
    
        print(f"Model '{model_name}' loaded from {self.full_path}")
        return model

    def _create_encoder(self, config):
        if config["type"] == "FCEncoder":
            return FCEncoder(config["one_hot_dim"], config["embedding_dim"])
        elif config["type"] == "CustomBertEncoder":
            return CustomBertEncoder(config["bert_model_name"], config["n_attention_layers"])
        elif config["type"] == "ImageEncodeur":
            return ImageEncodeur(weights=config["weights"], output_dim=config["embedding_dim"])
        elif config["type"] == "Identity":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown encoder type: {config['type']}")

    def _create_head(self, config):
        if config["type"] == "MLP":
            return MLP(config["concat_dim"], config["num_classes"])
        elif config["type"] == "SingleFC":
            return SingleFC(config["concat_dim"], config["num_classes"])
        else:
            raise ValueError(f"Unknown head type: {config['type']}")
        
    def construct_model(self, modalities):
        config = {}
        
