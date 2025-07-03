import torch
from utils.model_utils import get_mnist_loaders, train_model, FullyConnectedModel
from utils.visualization_utils import plot_training_history, plot_bar_data, plot_by_epoch


def compare_model_depth():
    """Сравнение моделей разной глубины"""

    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    times = []

    # 1 слой (линейный классификатор)
    linear_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2}
        ]
    )
    linear_history, linear_time = train_model(linear_model, train_loader, test_loader, epochs=5)
    times.append(linear_time)
    plot_training_history(linear_history, 'plots/linear_batch.png')
    plot_by_epoch(linear_history, 'plots/linear_epoch_batch.png')

    # 2 слоя (1 скрытый)
    two_layers_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"}  
        ]
    )
    two_layers_history,  two_layers_time = train_model(two_layers_model, train_loader, test_loader, epochs=5)
    times.append(two_layers_time)
    plot_training_history(two_layers_history, 'plots/two_layers_batch.png')
    plot_by_epoch(two_layers_history, 'plots/two_layers_epoch_batch.png')

    # 3 слоя (2 скрытых)
    three_layers_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"}
        ]
    )
    three_layers_history, three_layers_time = train_model(three_layers_model, train_loader, test_loader, epochs=5)
    times.append(three_layers_time)
    plot_training_history(three_layers_history, 'plots/three_layers_batch.png')
    plot_by_epoch(three_layers_history, 'plots/three_layers_epoch_batch.png')

    # 5 слоев (4 скрытых)
    five_layers_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 1024},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.1},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    )
    five_layers_history, five_layers_time = train_model(five_layers_model, train_loader, test_loader, epochs=5)
    times.append(five_layers_time)
    plot_training_history(five_layers_history, 'plots/five_layers_batch.png')
    plot_by_epoch(five_layers_history, 'plots/five_layers_epoch_batch.png')

    # 7 слоев (6 скрытых)
    seven_layers_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 1024},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"},
            {"type": "linear", "size": 768},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.1},
            {"type": "relu"},
            {"type": "linear", "size": 384},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.1},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "dropout", "rate": 0.1},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    )
    seven_layers_history, seven_layers_time = train_model(seven_layers_model, train_loader, test_loader, epochs=5)
    times.append(seven_layers_time)
    plot_training_history(seven_layers_history, 'plots/seven_layers_batch.png')
    plot_by_epoch(seven_layers_history, 'plots/seven_layers_epoch_batch.png')

    # График времени обучения моделей разной глубины
    models = ['linear', 'two-layer', 'three-layer', 'five-layer', 'seven-layer']
    plot_bar_data(times, models, 'Сравнение времени обучения моделей', 'plots/train_times.png')


if __name__ == '__main__':
    compare_model_depth()
