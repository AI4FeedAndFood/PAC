import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from TrainingTools.Training import MultiModalTrain, encode_train_test_columns
from TrainingTools.DataAndProcessor import get_train_test, ImageProcessor, TextProcessor, FlexibleMultiModalDataset
from DataProcess.RawDataPreprocess import set_config
from ModelTools.Manager import ModelManager

def load_config(config_path, version):
    return set_config(config_path, version)

def get_default_config():
    return {
        "used_columns": ["CleanDescription", "ProductCode"],
        "min_row": 5,
        "max_row": None,
        "batch_size": 64,
        "num_workers": 6,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "base_dim": 384,
        "modalities": {
                        'text':  {
                            'use_encoded': True,
                            'encoder': None,
                            'embedding_dim': 768
                        },
                        'client': {
                            'embedding_dim': 384
                            },
                        'laboratory': {
                            },
                        'label': {}
                    },

        "image_size": 224,
        "max_length": 512,
        "warmup_steps": 500,
        
    }

def main(folder_path, training_name, config_path, version):

    # Load configuration
    DATA_CONFIG = load_config(config_path, version)
    config = get_default_config()
    config.update(DATA_CONFIG)  # Override defaults with DATA_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get train and test data
    train, test = get_train_test(
        folder_path, 
        used_columns=config["used_columns"], 
        min_row=config["min_row"], 
        max_row=config["max_row"]
    )

    train, test = encode_train_test_columns(train, test, config["modalities"], training_name)

    # Initialize processors
    text_processor = TextProcessor()
    image_processor = ImageProcessor()

    # Create datasets
    train_dataset = FlexibleMultiModalDataset(train, config["modalities"], device, 
                                              image_processor=image_processor, 
                                              text_processor=text_processor)
    test_dataset = FlexibleMultiModalDataset(test, config["modalities"], device, 
                                             image_processor=image_processor, 
                                             text_processor=text_processor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                              num_workers=config["num_workers"], shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                            num_workers=config["num_workers"])

    # Initialize model
    model_manager = ModelManager(training_name)
    model = model_manager.construct_model(config["modalities"])

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=config["warmup_steps"], 
                                                num_training_steps=total_steps)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Get unique labels
    labels = train["ProductCode"].unique().tolist()

    # Train the model
    MultiModalTrain(model, train_loader, val_loader, optimizer, scheduler, loss_fn, 
                    config["num_epochs"], labels, training_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a multi-modal model")
    parser.add_argument("folder_path", type=str, help="Path to the data folder")
    parser.add_argument("training_name", type=str, help="Name for this training run")
    parser.add_argument("--config", type=str, default=r"Config\DataConfig.json", help="Path to configuration file")
    parser.add_argument("--version", type=str, default="VERSION_NG_FR", help="Configuration version")
    
    args = parser.parse_args()
    
    main(args.folder_path, args.training_name, args.config, args.version)