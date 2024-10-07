import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

class LinearHierarchicalLoss(nn.Module):
    def __init__(self, class_tree, class_count, depth_param=0.5, length_penalty=1.5):
        super().__init__()
        self.class_tree = class_tree
        self.length_penalty = length_penalty
        self.depth_param = depth_param

        # Calculer les poids inversement proportionnels à la fréquence
        max_count = max(list(class_count.values()))
        self.class_weights = {
            class_id: max_count / count
            for class_id, count in class_count.items()
        }

    def forward(self, y_pred, y_true):

        batch_size = y_pred.size(0)
        loss = 0

        for i in range(batch_size):
            pred_class = self.class_tree.get_matrix_from_id(y_pred[i].argmax().item())
            true_class = self.class_tree.get_matrix_from_id(y_true[i].argmax().item())  # Changé de argmax() à item()

            pred_path = self.class_tree.get_sub_pathes(pred_class)
            true_path = self.class_tree.get_sub_pathes(true_class)

            local_loss = 0
            common_depth = min(len(pred_path), len(true_path))
            for level in range(common_depth):
                # Calculer le poids λ(C(l))
                height = len(true_path) - level - 1
                level_weight = math.exp(-self.depth_param * height)

                pred_class_at_level = pred_path[level]
                true_class_at_level = true_path[level]

                pred_id = self.class_tree.get_matrix_id(pred_class_at_level)
                true_id = self.class_tree.get_matrix_id(true_class_at_level)

                # Créer un masque pour les classes valides à ce niveau
                valid_classes = set(self.class_tree.get_matrix_id(c) for c in
                                    self.class_tree.get_children('.'.join(true_class_at_level.split('.')[:level])))
                mask = torch.zeros_like(y_pred[i])
                mask[list(valid_classes)] = 1

                # Appliquer le masque et calculer la loss
                masked_pred = y_pred[i] * mask
                level_loss = F.cross_entropy(
                    masked_pred.unsqueeze(0),
                    torch.tensor([true_id], device=y_pred.device)
                )
                local_loss += level_weight * level_loss

            length_diff = abs(len(pred_path) - len(true_path))

            # Appliquer la pondération basée sur la fréquence
            class_weight = self.class_weights[true_class]

            # Combine tous les facteurs
            total_loss = local_loss * (self.length_penalty * length_diff) * class_weight

            loss += total_loss

        return loss / batch_size

class ConditionalHierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, class_tree, class_count, depth_param=0.1):
        super().__init__()
        self.class_tree = class_tree
        self.depth_param = depth_param

        # Calculer les poids inversement proportionnels à la fréquence
        max_count = max(class_count.values())
        self.class_weights = {
            class_id: max_count / count
            for class_id, count in class_count.items()
        }

    def forward(self, y_pred, y_true):
        # The model output is not softmaxed
        y_pred = F.softmax(y_pred, dim=1)

        batch_size = y_pred.size(0)
        loss = 0.0

        for i in range(batch_size):
            true_class = self.class_tree.get_matrix_from_id(y_true[i].argmax().item())
            true_path = self.class_tree.get_sub_pathes(true_class)

            path_loss = 0.0
            for level in range(len(true_path) - 1):
                current_path = true_path[level:]
                parent_path = true_path[level + 1:]

                # Calculer p(C(l)|C(l+1))
                current_indices = [self.class_tree.get_matrix_id(path) for path in current_path]
                parent_indices = [self.class_tree.get_matrix_id(path) for path in parent_path]

                current_prob = torch.sum(y_pred[i, current_indices])
                parent_prob = torch.sum(y_pred[i, parent_indices])

                conditional_prob = current_prob / (parent_prob + 1e-8)  # Éviter division par zéro

                # Calculer le poids λ(C(l)) selon l'équation (5)
                height = len(true_path) - level - 1
                weight = math.exp(-self.depth_param * height)

                # Ajouter à la loss selon l'équation (4)
                path_loss += weight * torch.log(conditional_prob + 1e-8)

            class_weight = self.class_weights[true_class]
            total_loss = class_weight * path_loss

            loss += - total_loss

        return loss / batch_size

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Calcule la perte contrastive pour un batch d'exemples.

        Args:
        - features: Tensor de forme (batch_size, embed_dim)
        - labels: Tensor de forme (batch_size,)

        Returns:
        - loss: La perte contrastive moyenne pour le batch
        """
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # Calcul de la matrice de similarité
        similarity_matrix = torch.matmul(features, features.T)
        
        # Normalisation pour la stabilité numérique
        sim_row_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - sim_row_max.detach()

        # Calcul des probabilités
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Calcul de la perte
        loss = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -loss.mean()

        return loss

def save_encoders_params(encoder, encoder_name, training_name, base_path="/content/drive/MyDrive/Data/PAC/models_weights"):
    """
    Save the encoder parameters to a specific folder.
    """
    # Create the full path to the folder
    full_path = os.path.join(base_path, training_name)

    # Create the folder if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    # Create the full path to the file
    joblib_file = os.path.join(full_path, encoder_name + '.pkl')

    # Save the encoder
    joblib.dump(encoder, joblib_file)

    print(f"Encoder '{encoder_name}' saved in {joblib_file}")

def fit_transform_onehot(train_col, min_frequency=1, categories="auto", test_col=None, label=False):
    """
    Fit and transform a column into one-hot encoded vectors.

    Args:
    train_col (list or array-like): Column to be encoded.
    min_frequency (int): Minimum frequency for categories.
    categories (str or list): Categories to use for encoding.
    test_col (list or array-like, optional): Column to be transformed using the fitted encoder.
    label (bool): Whether this is a label encoder.

    Returns:
    tuple: A tuple containing the fitted OneHotEncoder and the transformed train and test columns.
    """
    # Initialize OneHotEncoder
    if label:
        encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(categories=categories, sparse_output=False, min_frequency=min_frequency, handle_unknown="infrequent_if_exist")

    # Reshape train column
    train_col_reshaped = np.array(train_col).reshape(-1, 1)

    # Fit and transform train column
    train_encoded = encoder.fit_transform(train_col_reshaped)

    if test_col is not None:
        test_encoded = encoder.transform(np.array(test_col).reshape(-1, 1))
        return encoder, train_encoded, test_encoded

    return encoder, train_encoded

def encode_train_test_columns(train, test, modalities, training_name):
    """
    Encode columns in the dataframe based on the provided modalities.

    Args:
    train (DataFrame): The training dataframe.
    test (DataFrame): The test dataframe.
    modalities (dict): Dictionary defining the modalities and their encoding parameters.
    training_name (str): The name of the training session.

    Returns:
    DataFrame: The transformed training dataframe.
    DataFrame: The transformed test dataframe.
    """

    if "client" in modalities:
        client_oh_encoder, onehot_client_train, onehot_client_test = fit_transform_onehot(train["AccountCode"], test_col=test["AccountCode"])
        train["EncodedAccountCode"] = pd.Series(list(onehot_client_train)).values
        test["EncodedAccountCode"] = pd.Series(list(onehot_client_test)).values
        modalities["client"]['one_hot_encoder'] = client_oh_encoder
        modalities["client"]['one_hot_dim'] = onehot_client_train.shape[1]

        save_encoders_params(client_oh_encoder, "client", training_name)

    if "laboratory" in modalities:
        lab_oh_encoder, onehot_lab_train, onehot_lab_test = fit_transform_onehot(train["Laboratory"], test_col=test["Laboratory"])
        train["EncodedLaboratory"] = pd.Series(list(onehot_lab_train)).values
        test["EncodedLaboratory"] = pd.Series(list(onehot_lab_test)).values
        modalities["laboratory"]['onehot_dim'] = client_oh_encoder
        modalities["laboratory"]['embedding_dim'] = onehot_lab_train.shape[1]

        save_encoders_params(lab_oh_encoder, "laboratory", training_name)

    label_encoder, onehot_target_train, onehot_target_test = fit_transform_onehot(train["ProductCode"], categories=modalities["label"].get("categories", "auto"), test_col=test["ProductCode"], label=True)
    train["EncodedProductCode"] = pd.Series(list(onehot_target_train)).values
    test["EncodedProductCode"] = pd.Series(list(onehot_target_test)).values
    modalities["label"]['one_hot_encoder'] = client_oh_encoder
    modalities["label"]['num_classes'] = onehot_target_train.shape[1]

    save_encoders_params(label_encoder, "label", training_name)

    return train, test

def save_model_state_dict(model, model_name, training_name, base_path="/content/drive/MyDrive/Data/PAC/models_weights"):
    """
    Saves the model's state_dict in a specific folder.

    Args:
    model: The model object whose state_dict is to be saved
    model_name (str): Name of the model
    training_name (str): Name of the training session (used for folder name)
    base_path (str): Base path for saving the models
    """
    # Create the full path for the folder
    full_path = os.path.join(base_path, training_name)

    # Create the folder if it doesn't exist
    os.makedirs(full_path, exist_ok=True)

    # Create the full path for the file
    state_dict_file = os.path.join(full_path, f"{model_name}_state_dict.pth")

    # Save the model's state_dict
    torch.save(model.state_dict(), state_dict_file)

    print(f"Model state_dict for '{model_name}' saved in {state_dict_file}")

def MultiModalTrain(model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_epochs, labels, training_name):
    """
    Train the model for multiple epochs with validation.

    Args:
        model (MultiModalClassifier): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loss_fn (nn.Module): Loss function.
        num_epochs (int): Number of epochs to train the model.
        labels (list): List of label classes.
        training_name (str): Name for saving the best model.
        freeze_encoders (list): List of encoder names to freeze during training.

    Returns:
        tuple: Contains lists of training losses, validation losses, validation accuracies,
               learning rate history, best epoch, and best model.
    """
    
    best_model = model
    best_val_accuracy = 0.0
    best_epoch = 0
    train_losses, val_losses, val_accuracies = [], [], []
    learning_rate_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            labels = batch['label']

            # Compute loss
            computed_loss = loss_fn(outputs, labels)
            computed_loss.backward()
            optimizer.step()

            epoch_train_loss += computed_loss.item() * labels.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        learning_rate_history.append(optimizer.param_groups[0]['lr'])

        if epoch>10:
            # Validation loop
            model.eval()

            top3_accuracy = 0
            epoch_val_loss = 0.0
            val_predictions, val_actual_labels, raw_outputs = [], [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    outputs = model(val_batch)
                    labels = val_batch['label']

                    val_loss = loss_fn(outputs, labels)
                    epoch_val_loss += val_loss.item() * labels.size(0)

                    _, preds = torch.max(outputs, dim=1)
                    _, labels = torch.max(labels, dim=1)

                    raw_outputs.extend(outputs.cpu().tolist())
                    val_predictions.extend(preds.cpu().tolist())
                    val_actual_labels.extend(labels.cpu().tolist())

                    top3_pred = torch.topk(outputs, k=3, dim=1).indices
                    top3_accuracy += sum(labels[i] in top3_pred[i] for i in range(len(labels)))

            top3_accuracy /= len(val_loader.dataset)
            epoch_val_loss /= len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

            val_accuracy = accuracy_score(val_actual_labels, val_predictions)
            val_accuracies.append(val_accuracy)

            if scheduler:
                scheduler.step(val_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Learning rate: {learning_rate_history[-1]:.4f}: "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                print(f"-- Saving the best model {training_name} - acc: {round(best_val_accuracy, 4)} --\n")
                save_model_state_dict(model, "best_model", training_name)
                best_model = model

    return train_losses, val_losses, val_accuracies, learning_rate_history, best_epoch, best_model