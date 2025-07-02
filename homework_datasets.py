import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, categorical_data = None, numerical_data = None, binary_data = None):
        """
        Args:
            file_path - путь к файлу,
            target_column - целевой столбец,
            categorical_data - столбцы с категориальными данными,
            numerical_data - столбцы с числовыми данными,
            binary_data - столбцы с бинарными данными
        """

        self.file = pd.read_csv(file_path)
        self.data = pd.DataFrame()
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.file[target_column])

        # Обработка категориальных столбцов
        if categorical_data:
            self.categorical_encoders = {}
            for column in categorical_data:
                encoder = LabelEncoder()
                self.file[column] = encoder.fit_transform(self.file[column])
                self.categorical_encoders[column] = encoder
            self.data = pd.concat([self.data, self.file[categorical_data]], axis=1)

        # Обработка бинарных столбцов
        if binary_data:
            self.data = pd.concat([self.data, self.file[binary_data]], axis=1)

        # Обработка числительных столбцов
        if numerical_data:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(self.file[numerical_data])
            num_features = pd.DataFrame(scaled_values, columns=numerical_data)
            self.data = pd.concat([self.data, num_features], axis=1)

        self.X_numpy = self.data.values.astype(np.float32)
        self.y_numpy = self.y.astype(np.int64)
        
        self.X = torch.tensor(self.X_numpy, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y_numpy, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_tensor[idx]
    

def learning_logistic_regression():
    file_path = "data/plane_data.csv" 
    target_column = "satisfaction"

    # Определение типов столбцов
    categorical_data = ["Gender", "Customer Type", "Type of Travel", "Class"]
    numerical_data = [
        "Age", "Flight Distance", 
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]

    # Загрузка данные
    dataset = CSVDataset(
        file_path=file_path,
        target_column=target_column,
        categorical_data=categorical_data,
        numerical_data=numerical_data
    )

    # Разделение на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.2, random_state=42
    )

    # Обучение логистической регрессии
    model = LogisticRegression(max_iter=1000)  # Увеличим max_iter для сходимости
    model.fit(X_train, y_train)

    # Предсказание и оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def learning_linear_regression():
    dataset = CSVDataset(
        file_path='data/sales_data.csv',
        target_column='Demand',
        categorical_data=['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'],
        numerical_data=['Inventory Level', 
                        'Units Sold', 'Units Ordered', 
                        'Price', 'Discount', 
                        'Competitor Pricing']
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Обучение модели линейной регрессии
    model = LinearRegression()

    X_list = []
    y_list = []

    for X_batch, y_batch in dataloader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)

    model.fit(X_all, y_all)

    # Оценка
    predictions = model.predict(X_all)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_all, predictions)
    print(f"Mean Squared Error: {mse}")


if __name__ == '__main__':
    learning_linear_regression()
    learning_logistic_regression()