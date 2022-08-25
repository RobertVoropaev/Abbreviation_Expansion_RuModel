import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvResNet(nn.Module):
    def __init__(self, input_size, num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    input_size, 
                    input_size, 
                    kernel_size, 
                    padding=kernel_size//2 
                ),
                nn.Dropout(dropout),
                nn.LeakyReLU()
            )
            layers.append(conv_block)
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = x + layer(x) 
        x = x.permute(0, 2, 1)
        return x

class RnnNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, 
                 batch_first=True, dropout=0.1, bidirectional=False):
        super().__init__()
        if bidirectional:
            hidden_size = hidden_size // 2
        
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
    def forward(self, x):
        x = self.rnn(x)
        return x[0]
    
class NnClassifier(nn.Module): 
    def __init__(self, tokens_num, labels_num, embedding_size=64, 
                 backbone=ConvResNet, **kwargs):
        super().__init__()
        self.embeddings = nn.Embedding(tokens_num, embedding_size, padding_idx=0)
        
        self.backbone = backbone(**kwargs)
        
        self.out = nn.Linear(embedding_size, labels_num)
    
    def forward(self, tokens):
        batch_size, max_sent_len = tokens.shape

        embeddings = self.embeddings(tokens)
        
        features = self.backbone(embeddings)
        logits = self.out(features)
        
        return logits.permute(0, 2, 1)