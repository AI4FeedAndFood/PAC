import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel

from torchvision import transforms, models

import torch.nn.functional as F

# Models dedicated to images
class MultiImageAttentionEncodeur:
    def __init__(self, num_classes=8000, embed_dim=2100):
        super(MultiImageAttentionEncodeur, self).__init__()
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

# Models dedicated to client   
class FCEncoder(nn.Module):
    def __init__(self, one_hot_dim, output_dim):
        super(FCEncoder, self).__init__()
        self.embedding = nn.Linear(in_features=one_hot_dim, out_features=output_dim, bias=True)

    def forward(self, encoded_client):
        return self.embedding(encoded_client)

# Models dedicated to text
class CustomBertEncoder(nn.Module):
    """
    A model based on BERT with mean pooling and a classification head.

    Attributes:
        bert (AutoModel): The BERT model used for encoding input text.
        classifier (nn.Module): The classification head.
    """
    def __init__(self, bert_model_name, n_attention_layers=None):
        super(CustomBertEncoder, self).__init__()
        self.CustomBertEncoder = AutoModel.from_pretrained(bert_model_name)
        if n_attention_layers:
            self.CustomBertEncoder.encoder.layer = self.CustomBertEncoder.encoder.layer[:n_attention_layers]

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the token embeddings.

        Args:
            model_output (tuple): The output from BERT model, contains token embeddings.
            attention_mask (torch.Tensor): Tensor indicating which tokens are real words (1) and which are padding (0).

        Returns:
            torch.Tensor: Mean pooled sentence embeddings.
        """
        token_embeddings = model_output.last_hidden_state #token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, tokenized_text):
        """
        Forward pass of the SBERT model.

        Args:
            tokenized_text (dict): Dictionary containing input_ids and attention_mask.

        Returns:
            torch.Tensor: Logits for each class.
        """

        model_output = self.CustomBertEncoder(**tokenized_text)
        sentence_embeddings = self.mean_pooling(model_output, tokenized_text["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def encode(self, tokenized_text, device='cpu'):
        """
        Encode a tokenized text input using the BERT model.

        Args:
            tokenized_text (dict): The tokenized text input with keys 'input_ids' and 'attention_mask'.
            device (str): Device to perform encoding on (CPU or GPU).

        Returns:
            torch.Tensor: Encoded text embeddings.
        """
        self.to(device)
        self.eval()

        with torch.no_grad():
            # Move tokenized inputs to the specified device
            tokenized_text = {key: value.to(device) for key, value in tokenized_text.items()}

            # Get the embeddings
            embeddings = self.forward(tokenized_text)

        return embeddings.cpu().squeeze()

# Models dedicated to be pluged at the head of a multi-modal NN
class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for classification.

    Attributes:
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Sequential): A multi-layer perceptron (MLP) for classification.
    """
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim//2),
            nn.Dropout(0.1),
            nn.Linear(input_dim//2, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.bn(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

class SingleFC(nn.Module):
    """
    A single fully connected layer.

    Attributes:
        classifier (nn.Sequential): A fully-connected layer.
    """
    def __init__(self, input_dim, num_classes):
        super(SingleFC, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits for each class.
        """
        logits = self.classifier(x)
        return logits

# Model to handle specific modules
class MultiModalClassifier(nn.Module):
    def __init__(self, modalities, head):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.modalities = modalities

        # Initialiser les encodeurs en fonction des modalités disponibles
        if 'text' in modalities:
            if modalities['text'].get('use_encoded', True):
                self.encoders['text'] = nn.Identity()  # Pas besoin d'encoder, déjà fait
            else:
                self.encoders['text'] = modalities['text']['encoder']

        if 'client' in modalities:
            self.encoders['client'] = modalities['client']

        if 'image' in modalities:
            self.encoders['image'] = modalities['image']

        self.head = head

    def forward(self, batch):
        features = []


        if 'laboratory' in self.modalities and 'laboratory' in batch:
            features.append(batch['laboratory'])

        if 'client' in self.modalities and 'client' in batch:
            features.append(self.encoders['client'](batch['client']))

        if 'text' in self.modalities and 'text' in batch:
            features.append(self.encoders['text'](batch['text']))

        if 'image' in self.modalities and 'image' in batch:
            features.append(self.encoders['image'](batch['image']))

        if not features:
            raise ValueError("Aucune modalité valide n'a été fournie dans le batch.")

        combined = torch.cat(features, dim=1)

        return self.head(combined)

class MultiModalAttention(nn.Module):
    def __init__(self, dim, scale_ratio=-0.5):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** scale_ratio

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        return attn @ v

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.norm = nn.LayerNorm(input_dim)
        self.attention = MultiModalAttention(input_dim)
        self.projection = nn.Linear(input_dim, self.output_dim) if input_dim != self.output_dim else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.attention(x)
        x = self.projection(x)
        return self.batch_norm(x)

class FusionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, output_dim)
        self.norm1 = nn.BatchNorm1d(input_dim // 2)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)
        return self.dropout(x)

class MultiModalAttentionClassifier(nn.Module):
    def __init__(self, image_dim=2100, text_dim=800, client_dim=400, lab_dim=3, num_classes=7000, use_image=True):
        super().__init__()
        
        self.use_image = use_image
        
        if self.use_image:
            self.image_encoder = ModalityEncoder(image_dim)
        
        self.text_encoder = ModalityEncoder(text_dim)
        self.client_encoder = ModalityEncoder(client_dim)
        self.lab_encoder = ModalityEncoder(lab_dim)
        
        total_dim = text_dim + client_dim + lab_dim
        if self.use_image:
            total_dim += image_dim
        
        self.fusion = nn.Sequential(
            FusionBlock(total_dim, total_dim // 2),
            FusionBlock(total_dim // 2, total_dim // 4),
            FusionBlock(total_dim // 4, total_dim // 8)
        )
        
        self.classifier = nn.Linear(total_dim // 8, num_classes)

    def forward(self, text, client, lab, image=None):
        if self.use_image and image is None:
            raise ValueError("Image input is required when use_image is True")
        
        features = []
        
        if self.use_image:
            image_feat = self.image_encoder(image)
            features.append(image_feat)
        
        text_feat = self.text_encoder(text)
        client_feat = self.client_encoder(client)
        lab_feat = self.lab_encoder(lab)
        
        features.extend([text_feat, client_feat, lab_feat])
        
        combined = torch.cat(features, dim=1)
        fused = self.fusion(combined)
        
        logits = self.classifier(fused)
        return logits
