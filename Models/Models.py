import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()

        self.two_dense_layers = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=512, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=512, out_features=num_classes, bias=True),
        nn.Softmax(dim=1)
    )

    def forward(self, text_features):

        # Pass through dense layers
        x = self.two_dense_layers(text_features)

        return x