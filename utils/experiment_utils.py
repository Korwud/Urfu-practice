import time
import torch.optim as optim
import torch.nn as nn
from utils.model_utils import run_epoch
import pandas as pd

def train_model_l2(model, train_loader, test_loader, epochs=5, l2_weigth=0.01, device='cpu'):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=l2_weigth)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)
    
    end_time = time.time()
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }, round(end_time - start_time, 4)


def analyze_layers(model):
    layer_stats = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_stats.append({
                "layer": name,
                "mean": param.mean().item(),
                "std": param.std().item(),
                "sparsity": (param == 0).float().mean().item()
            })
    return pd.DataFrame(layer_stats)