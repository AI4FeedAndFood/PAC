import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image

class ProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = [Image.open(path) for path in self.image_paths[idx]]
        if self.transform:
            images = [self.transform(img) for img in images]
        return torch.stack(images), self.labels[idx]

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Créer le dataset et le dataloader
# Vous devrez remplacer ces listes par vos propres données
image_paths = [['path1_1.jpg', 'path1_2.jpg'], ['path2_1.jpg'], ...]
labels = [0, 1, ...]  # Vos labels de 0 à 7999
dataset = ProductDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

class ImageEncoder(nn.Module):
    def __init__(self, num_classes=8000, embed_dim=2100):
        super(ImageEncoder, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.projection = nn.Linear(2048, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, num_images, c, h, w = x.size()
        x = x.view(batch_size * num_images, c, h, w)
        
        features = self.base_model(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        
        embeddings = embeddings.view(batch_size, num_images, -1)
        embeddings = embeddings.permute(1, 0, 2)
        
        attn_output, _ = self.attention(embeddings, embeddings, embeddings)
        attn_output = attn_output.mean(dim=0)
        
        logits = self.classifier(attn_output)
        return attn_output, logits

model = ImageEncoder()

def contrastive_loss(features, labels, temperature=0.07):
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    similarity_matrix = torch.matmul(features, features.T)
    
    # Normalisation pour la stabilité numérique
    sim_row_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    similarity_matrix = similarity_matrix - sim_row_max.detach()

    # Calculer les pertes
    exp_sim = torch.exp(similarity_matrix / temperature)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    loss = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    loss = -loss.mean()
    return loss

criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_image(model, train_loader, val_loader, num_epochs, device, lr=0.001, alpha=0.5):
    """
    Fonction pour entraîner le modèle d'image.

    Args:
    - model: Le modèle ProductEncoder
    - train_loader: DataLoader pour les données d'entraînement
    - val_loader: DataLoader pour les données de validation
    - num_epochs: Nombre d'époques d'entraînement
    - device: Device sur lequel effectuer l'entraînement (cuda ou cpu)
    - lr: Taux d'apprentissage
    - alpha: Coefficient pour équilibrer les pertes contrastive et de classification

    Returns:
    - model: Le modèle entraîné
    - history: Dictionnaire contenant l'historique des pertes et des métriques
    """

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Boucle d'entraînement
        for batch_images, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            embeddings, logits = model(batch_images)
            
            loss_contrastive = contrastive_loss(embeddings, batch_labels)
            loss_cls = criterion_cls(logits, batch_labels)
            
            loss = alpha * loss_contrastive + (1 - alpha) * loss_cls
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                embeddings, logits = model(batch_images)
                
                loss_contrastive = contrastive_loss(embeddings, batch_labels)
                loss_cls = criterion_cls(logits, batch_labels)
                loss = alpha * loss_contrastive + (1 - alpha) * loss_cls

                val_loss += loss.item()

                _, predicted = logits.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Ici, vous pouvez ajouter une logique pour sauvegarder le meilleur modèle
        # Par exemple :
        # if val_accuracy > best_accuracy:
        #     best_accuracy = val_accuracy
        #     torch.save(model.state_dict(), 'best_model.pth')

    return model, history

# Utilisation de la fonction

if __name__ == "__main__":
    # Assurez-vous que ces éléments sont définis ailleurs dans votre code
    model = ImageEncoder()
    train_loader = DataLoader(...)  # Votre DataLoader d'entraînement
    val_loader = DataLoader(...)    # Votre DataLoader de validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model, training_history = train_image(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        device=device,
        lr=0.001,
        alpha=0.5
    )

    # Vous pouvez maintenant utiliser trained_model et training_history
    # pour d'autres tâches, comme la visualisation des résultats ou
    # l'intégration dans votre pipeline multi-modal
