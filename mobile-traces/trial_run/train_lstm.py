import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
import copy
import random

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        z = torch.cat([last, static], dim = 1)
        return self.classifier(z).squeeze(1)

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

def handle_data(stage, strat, raw_path, labels_path):
    K, target = STAGES[stage]['K'], STAGES[stage]['target']
    raw = pd.read_csv(raw_path)
    labels = pd.read_csv(labels_path)

    # cut to the first K seconds + attach lables
    window = raw[raw['k'] <= K]
    df = window.merge(labels, on='id') # [CHANGE ME]
    if stage == 2:
        df = df[df['is_bar'] == 0]
    df = df.sort_values(['id', 'k'])

    rows_per_file = df.groupby('id').size()
    assert (rows_per_file == K).all(), "Time length per match did not match"

    # Do per-second change or cumulative counters?
    # per-second change 
    assert strat in ['delta', 'cumulative'], "strat should be delta or comulative"
    if strat == 'delta':
        feats = df.groupby('id')[COLS].diff().fillna(df[COLS]) # Calculate diff, first entry is na, so, fill with original values
    elif strat == 'cumulative':
        feats = df[COLS]
    n_files = len(rows_per_file)
    X = np.log1p(feats.to_numpy(np.float32)).reshape(n_files, K, len(COLS)) 

    per_file = df.groupby('id', sort=True).agg(
        file_size=('file_size', 'max'),
        y=(target, 'first'),
        split=('split', 'first')
    )
    static = np.log1p(per_file[['file_size']].to_numpy(np.float32))

    y = per_file['y'].to_numpy(np.float32)
    split = per_file['split'].to_numpy()
    return X, static, y, split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    X, static, y, split = handle_data(
        stage = args.stage,
        strat = 'delta',
        raw_path = 'raw_dummy.csv',
        labels_path = 'labels.csv'
    )
    idx = {
        s: split == s for s in SPLITS
    }
    print(f"""\
Stage {args.stage}, X {X.shape}
{' '.join(f'{s} {idx[s].sum()} (pos {int(y[idx[s]].sum())})' for s in SPLITS)}
""")

    # normalize with train statistics
    x_mean = X[idx['train']].reshape(-1, X.shape[2]).mean(0)
    x_std  = X[idx['train']].reshape(-1, X.shape[2]).std(0) + 1e-6 # just in case std is 0
    X = (X - x_mean)/x_std
    s_mean, s_std = static[idx['train']].mean(0), static[idx['train']].std(0) + 1e-6
    static = (static - s_mean)/s_std

    t = {
        s: (torch.tensor(X[idx[s]]), torch.tensor(static[idx[s]]), torch.tensor(y[idx[s]])) for s in SPLITS
    }
    train_dl = DataLoader(
        TensorDataset(*t['train']),
        batch_size=16,
        shuffle=True
    )

    # (self, input_size, hidden_size, num_layers, n_static, output_size = 1):
    model = LSTMClassifier(
        input_size = X.shape[2],
        hidden_size = args.hidden,
        num_layers = args.layers,
        n_static = static.shape[1]
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Class is imbalanced, I'll just use simple class weighting for now,
    # but, let's think about this more later on
    n_pos = y[idx['train']].sum()
    pos_weight = torch.tensor((len(y[idx['train']]) - n_pos)/max(n_pos, 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def evaluate(split_name):
        model.eval()
        xb, sb, yb = t[split_name]
        with torch.no_grad():
            logits = model(xb, sb)
        return loss_fn(logits, yb).item(), metrics(logits, yb)

    best_loss, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, sb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping, standard for LSTM
            # if vanishing gradient happen, add residual stream to LSTM
            opt.step()

        val_loss, val_metric = evaluate("val")
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train loss = {evaluate('train')[0]:.3f}")
            print(f"                 | val loss   = {val_loss:.3f} | (acc, prec, recall, f1): {val_metric}")

    model.load_state_dict(best_state)
    _, test_m = evaluate('test')
    majority_baseline_logits = max(y[idx['test']].mean(), 1 - y[idx['test']].mean())

    print(f"""\
Test             : "{" ".join(f"{k} {v:.2f}" for k, v in test_m.items())}
Majority Accuracy: {majority_baseline_logits:.3f}
""")

if __name__ == "__main__":
    main()