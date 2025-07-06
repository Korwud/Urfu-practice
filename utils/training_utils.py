from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time


class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFARDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_cifar_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFARDataset(train=True, transform=transform)
    test_dataset = CIFARDataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False, track_gradients=False):
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    gradients = {} if track_gradients else None
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        
        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return (total_loss / len(data_loader), correct / total), gradients


def train_model(model, train_loader, test_loader, epochs=3, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    all_gradients = {}
    
    start_time = time.time()
    for epoch in range(epochs):
        (train_loss, train_acc), epoch_gradients = run_epoch(
            model, train_loader, criterion, optimizer, device, 
            is_test=False, track_gradients=True
        )
        for name, value in epoch_gradients.items():
            all_gradients[name] = value
        
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, None, device, is_test=True
        )[0]
        
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
        'test_accs': test_accs,
        'gradients': all_gradients 
    }, round(end_time - start_time, 4)


def evaluate_model(model, train_loader, test_loader, epochs=3, device='cpu'):
    history, learning_time = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    cm = confusion_matrix(all_targets, all_preds)
    
    start_time = time.time()
    with torch.no_grad():
        _ = run_epoch(model, test_loader, nn.CrossEntropyLoss(), None, device, is_test=True)
    inference_time = round(time.time() - start_time, 4)
    
    parameters_count = count_parameters(model)
    
    return {
        'history': history,
        'learning_time': learning_time,
        'inference_time': inference_time,
        'parameters_count': parameters_count,
        'accuracy': max(history['test_accs']),
        'confusion_matrix': cm, 
        'gradients': history['gradients']  
    }

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)