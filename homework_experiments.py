import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import r2_score
import pandas as pd

def hyper_parameters_research():
    """
    Функция для проверки различных гиперпараметров
    """

    # 1. Загрузка и подготовка данных
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Класс линейной регрессии
    class LinearRegression(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)

    # 3. Параметры экспериментов
    learning_rates = [0.01, 0.1]
    batch_sizes = [32, 64]
    optimizers = {
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }

    # 4. Проведение экспериментов
    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt_name, opt_class in optimizers.items():
                # Подготовка данных
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train), 
                    torch.FloatTensor(y_train).unsqueeze(1)
                )
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Модель и оптимизатор
                model = LinearRegression(X_train.shape[1])
                criterion = nn.MSELoss()
                optimizer = opt_class(model.parameters(), lr=lr)
                
                # Обучение
                losses = []
                for epoch in tqdm(range(100), desc=f'lr={lr}, bs={batch_size}, opt={opt_name}'):
                    epoch_loss = 0
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    losses.append(epoch_loss / len(train_loader))
                
                # Оценка на тесте
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test))
                    test_mse = criterion(y_pred, torch.FloatTensor(y_test).unsqueeze(1)).item()
                
                results.append({
                    'optimizer': opt_name,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'train_loss': losses[-1],
                    'test_mse': test_mse,
                    'loss_history': losses
                })

    # 5. Визуализация результатов
    plt.figure(figsize=(15, 10))

    # Графики сходимости
    for i, res in enumerate(results):
        plt.subplot(3, 3, i+1)
        plt.plot(res['loss_history'])
        plt.title(f"{res['optimizer']}, lr={res['learning_rate']}, bs={res['batch_size']}")
        plt.xlabel('Epoch')
        plt.ylabel('MSE')

    plt.tight_layout()
    plt.savefig('plots/hyper_parameteres.png')
    plt.show()


def feature_engineering():
    """
    Функция для исследования методов feature engineering
    """
    # 1. Загрузка и подготовка данных
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Класс линейной модели
    class LinearModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)

    # 3. Функция для обучения и оценки
    def train_evaluate(X_train, X_test, y_train, y_test, epochs=100, lr=0.01):
        # Конвертация в тензоры
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
        
        model = LinearModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Обучение
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        
        # Оценка
        with torch.no_grad():
            y_pred = model(X_test_t).numpy()
            r2 = r2_score(y_test, y_pred)
            mse = criterion(model(X_test_t), y_test_t).item()
        
        return r2, mse

    # 4. Создание новых признаков
    # Базовая модель (исходные признаки)
    base_r2, base_mse = train_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    # a. Полиномиальные признаки
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_train_poly, X_test_poly = train_test_split(X_poly, test_size=0.2, random_state=42)
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)
    poly_r2, poly_mse = train_evaluate(X_train_poly, X_test_poly, y_train, y_test)

    # b. Статистические признаки
    X_stats = np.column_stack((
        X,
        np.mean(X, axis=1, keepdims=True),
        np.std(X, axis=1, keepdims=True),
        np.median(X, axis=1, keepdims=True)
    ))
    X_train_stats, X_test_stats = train_test_split(X_stats, test_size=0.2, random_state=42)
    X_train_stats = scaler.fit_transform(X_train_stats)
    X_test_stats = scaler.transform(X_test_stats)
    stats_r2, stats_mse = train_evaluate(X_train_stats, X_test_stats, y_train, y_test)

    # 5. Сравнение результатов
    results = [
        {'Method': 'Base', 'R2': base_r2, 'MSE': base_mse},
        {'Method': 'Polynomial', 'R2': poly_r2, 'MSE': poly_mse},
        {'Method': 'Statistical', 'R2': stats_r2, 'MSE': stats_mse}
    ]
    
    df_results = pd.DataFrame(results)
    print("\nFeature Engineering Results:")
    print(df_results)

    # 6. Визуализация
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(df_results['Method'], df_results['R2'])
    plt.title('R2 Score Comparison')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.bar(df_results['Method'], df_results['MSE'])
    plt.title('MSE Comparison')
    
    plt.tight_layout()
    plt.savefig('plots/feature_engineering.png')
    plt.show()


if __name__ == '__main__':
    hyper_parameters_research()
    feature_engineering()
