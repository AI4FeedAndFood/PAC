import torch
import torch.nn.functional as F
from collections import defaultdict

from DataProcess.RawDataPreprocess import clean_description, set_config
from ModelTools.Manager import ModelManager

version = "VERSION_NG_FR"
config_path = r"Config\DataConfig.json"
DATA_CONFIG = set_config(config_path, version)


def evaluate(model, data_loader, device, product_codes, valid_df):
    model.eval()
    accuracies = defaultdict(float)
    total = 0
    top_3_predictions = []

    with torch.no_grad():
        for batch_idx, val_batch in enumerate(data_loader):
            outputs = model(val_batch)
            outputs = F.softmax(outputs, dim=1)
            labels = val_batch['label'].to(device)
            
            # Calculate accuracies for this batch
            _, top1_pred = torch.max(outputs, dim=1)
            top3_pred = torch.topk(outputs, k=3, dim=1).indices
            top5_pred = torch.topk(outputs, k=5, dim=1).indices

            _, labels = torch.max(labels, dim=1)
            
            accuracies['top1'] += (top1_pred == labels).sum().item()
            accuracies['top3'] += sum(labels[i] in top3_pred[i] for i in range(len(labels)))
            accuracies['top5'] += sum(labels[i] in top5_pred[i] for i in range(len(labels)))
            
            total += labels.size(0)
            
            # Get top 3 predictions for each sample in the batch
            batch_top3 = torch.topk(outputs, k=3, dim=1)
            for i in range(len(labels)):
                top_3_predictions.append({
                    'Top1_Product': product_codes[batch_top3.indices[i, 0].item()],
                    'Top2_Product': product_codes[batch_top3.indices[i, 1].item()],
                    'Top3_Product': product_codes[batch_top3.indices[i, 2].item()],
                    'Top1_Prob': batch_top3.values[i, 0].item(),
                    'Top2_Prob': batch_top3.values[i, 1].item(),
                    'Top3_Prob': batch_top3.values[i, 2].item()
                })
            
            # Clear unnecessary tensors to free up memory
            del outputs, labels, top1_pred, top3_pred, top5_pred, batch_top3
            torch.cuda.empty_cache()


    # Calculate final accuracies
    for k in accuracies:
        accuracies[k] /= total

    print(f"Top-1 Accuracy: {accuracies['top1']:.6f}")
    print(f"Top-3 Accuracy: {accuracies['top3']:.6f}")
    print(f"Top-5 Accuracy: {accuracies['top5']:.6f}")

    # Add top-3 predictions to valid_df
    for i, pred in enumerate(top_3_predictions):
        for k, v in pred.items():
            valid_df.at[i, k] = v

    return accuracies, valid_df

def main(training_name, eval_df):

    # Load models
    model_manager = ModelManager(training_name)
    model = model_manager.load_model("best_model")


    # Process eval data
    eval_df = clean_description(eval_df, DATA_CONFIG["LANG"], DATA_CONFIG["DEL_LIST"], DATA_CONFIG["CONVERT_DICT"])
    
    # Run evaluation
    accuracies, updated_valid_df = evaluate(model, valid_loader, device, product_codes, valid_df)