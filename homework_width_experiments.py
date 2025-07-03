import torch
from utils.model_utils import FullyConnectedModel, train_model, get_mnist_loaders, count_parameters
from utils.visualization_utils import plot_bar_data, draw_heatmap


def compare_model_width():
    """Сравнение моделей разной ширины"""

    train_loader, test_loader = get_mnist_loaders()
    times = []
    max_accurancies = []

    # Узкие слои: [64, 32, 16]
    narrow_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 64},
            {"type": "relu"},
            {"type": "linear", "size": 32},
            {"type": "relu"},
            {"type": "linear", "size": 16},
            {"type": "relu"}
        ]
    )
    narrow_history, narrow_time = train_model(narrow_model, train_loader, test_loader, epochs=5)
    times.append(narrow_time)
    max_accurancies.append(max(narrow_history['train_accs']))

    # Средние слои: [256, 128, 64]
    medium_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"},
            {"type": "linear", "size": 64},
            {"type": "relu"}
        ]
    )
    medium_history, medium_time = train_model(medium_model, train_loader, test_loader, epochs=5)
    times.append(medium_time)
    max_accurancies.append(max(medium_history['train_accs']))

    # Широкие слои: [1024, 512, 256]
    wide_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"}
        ]
    )
    wide_history, wide_time = train_model(wide_model, train_loader, test_loader, epochs=5)
    times.append(wide_time)
    max_accurancies.append(max(wide_history['train_accs']))

    # Очень широкие слои: [2048, 1024, 512]
    xwide_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 2048},
            {"type": "relu"},
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"}
        ]
    )
    xwide_history, xwide_time = train_model(xwide_model, train_loader, test_loader, epochs=5)
    times.append(xwide_time)
    max_accurancies.append(max(xwide_history['train_accs']))

    models = ['narrow', 'medium', 'wide', 'xwide']
    parameters_count = [count_parameters(narrow_model), count_parameters(medium_model),
                         count_parameters(wide_model), count_parameters(xwide_model)]
    plot_bar_data(times, models, 'Сравнение времени обучения моделей', 'plots/width_train_times.png')
    plot_bar_data(max_accurancies, models, 'Сравнение точности моделей', 'plots/width_acc.png')
    plot_bar_data(parameters_count, models, 'Сравнение количества параметров моделей', 'plots/parameteres_count.png')


def architecture_optimization():
    """Поиск оптимальной архитектуры с помощью grid search"""
    train_loader, test_loader = get_mnist_loaders()
    width_schemes = {
        "Расширение": [128, 256, 512],
        "Сужение": [512, 256, 128],
        "Постоянная": [256, 256, 256],
    }

    results = []
    for name, widths in width_schemes.items():
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=[
                {"type": "linear", "size": widths[0]},
                {"type": "relu"},
                {"type": "linear", "size": widths[1]},
                {"type": "relu"},
                {"type": "linear", "size": widths[2]},
                {"type": "relu"}
            ]
        )
        history, time = train_model(model, train_loader, test_loader, epochs=5)
        test_acc = max(history['train_accs'])
        results.append({"Scheme": name, "Widths": '-'.join(map(str, widths)), "Train Acc": test_acc})

    draw_heatmap(results, 'plots/heatmap.png')


if __name__ == '__main__':
    #compare_model_width()
    architecture_optimization()