import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import make_regression_data, log_epoch, RegressionDataset, make_classification_data, accuracy, ClassificationDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def modify_linear_regression():
    """
    Модифицированная функция линейной регрессии с регуляризацией и ранней остановкой
    """

    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и делим на обучающую и валидационную выборки
    dataset = RegressionDataset(X, y)
    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f'Размер обучающего набора: {len(train_dataset)}')
    print(f'Размер валидационного набора: {len(val_dataset)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Параметры регуляризации
    lambda_L1 = 0.001
    lambda_L2 = 0.001
    
    # Параметры ранней остановки
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            
            # Добавляем регуляризацию к функции потери
            params = list(model.parameters())
            l1_penalty = sum(torch.sum(torch.abs(p)) for p in params)
            l2_penalty = sum(torch.sum(p ** 2) for p in params)
            loss_with_reg = loss + lambda_L1 * l1_penalty + lambda_L2 * l2_penalty
            
            loss_with_reg.backward()
            optimizer.step()
            
            total_loss += loss_with_reg.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Валидация для ранней остановки
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_pred = model(val_X)
                val_loss_total += criterion(val_pred, val_y).item()
        avg_val_loss = val_loss_total / len(val_loader)
        
        log_epoch(epoch, avg_train_loss)
        print(f"Epoch {epoch}: Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping", best_loss)
                break



class LogisticRegression(nn.Module):

    # Добавлена многоклассовая классификация
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def modify_logistic_regression():
    """
    Модифицированная функция логистической регрессии, добавлена многоклассовая
    классификация, визуализация и отображение различных параметров
    """
    # Генерируем данные
    X, y = make_classification_data(n=200, num_classes=3)
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель для многоклассовой классификации, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=3, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        all_preds = []
        all_targets = []
        all_logits = []
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            batch_y = batch_y.squeeze().long()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            logits = model(batch_X)
            all_logits.append(logits.detach().cpu())
        
        avg_loss = total_loss / (i + 1)
        
        # Метрики для всей эпохи
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        all_logits = torch.cat(all_logits)
        probas = torch.softmax(all_logits, dim=1).numpy()
        roc_auc = roc_auc_score(np.array(all_targets), probas, multi_class='ovr')
        

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Визуализируем confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    plt.figure(figsize=(8,6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('plots/logistic.png')
    plt.show()
    

    # Сохраняем модель
    torch.save(model.state_dict(), 'models/logreg_torch.pth')
    
    # Загружаем модель
    new_model = LogisticRegression(in_features=3, num_classes=3)
    new_model.load_state_dict(torch.load('models/logreg_torch.pth'))
    new_model.eval() 

if __name__ == '__main__':
    modify_linear_regression()
    modify_logistic_regression()