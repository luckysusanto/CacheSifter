import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensoerDataset

STAGES = {
    1: {'K': 20, 'target': 'is_bar'},
    2: {'K': 60, 'target': 'is_long'}
}
COLS = [
    'read_amount',
    'read_count',
    'write_amount',
    'write_count'
]
SPLITS = [
    'train', 'val', 'test'
]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_static, output_size = 1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size + n_static, output_size) # n_static for features 

    def forward(self, x, static):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device())
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device())
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        z = torch.cat([last, static], dim = 1)
        return self.fc(z).squeeze(1)

def metrics(logits, y):
    pred = (logits > 0).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    # tn = ((pred == 0) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'acc': (pred == y).float().mean().item(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def handle_data(stage, raw_path, labels_path):
    K, target = STAGES[stage]['K'], STAGES[stage]['target']
    raw = pd.read_csv(raw_path)
    labels = pd.read_csv(labels_path)

    # cut to the first K seconds + attach lables
    window = raw[raw['k'] <= K]
    df = window.merge(labels, on='id')
    pass