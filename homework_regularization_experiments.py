import torch
from utils.model_utils import FullyConnectedModel, train_model, get_mnist_loaders
from utils.experiment_utils import train_model_l2, analyze_layers
from utils.visualization_utils import draw_learning_stability, draw_weigths, draw_adaptive_regularization


def compare_regularization_technics():
    """Сравнивает различные техники регуляризации"""
    train_loader, test_loader = get_mnist_loaders()

    config = {
        'Без регуляризации': [
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ],

        'Только Dropout': [
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.5},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.3},
            {"type": "linear", "size": 128},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.1}
        ],

        'Только BatchNorm': [
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "relu"}
        ],

        'Комбинированная модель': [
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.5},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.3},
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.1}
        ],

        'L2 модель': [
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ] 
    }

    results = []
    for name, layers in config.items():
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=layers
        )
        
        if name == 'L2 модель':
            history, time = train_model_l2(model, train_loader, test_loader)
        else:
            history, time = train_model(model, train_loader, test_loader)
        
        weights = []
        for param in model.parameters():
            if param.dim() > 1:
                weights.extend(param.view(-1).cpu().detach().numpy())

        results.append({
            "Config": name,
            "Train Acc": max(history["train_accs"]),
            "Weights": weights,
            "History": history
        })

    draw_weigths(results, 'plots/weigths.png')
    draw_learning_stability(results, 'plots/stability.png')

def adaptive_regularization():
    """Сравнивает виды адаптивной регуляризации"""

    train_loader, test_loader = get_mnist_loaders()
    configs = {
        "Adaptive Dropout": [
            {"type": "linear", "size": 512},
            {"type": "adaptive_dropout", "max_rate": 0.5},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"}
        ],
        
        "Variable BatchNorm": [
            {"type": "linear", "size": 512},
            {"type": "batch_norm", "momentum": 0.1},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm", "momentum": 0.3},
            {"type": "relu"}
        ],
        
        "Combined": [
            {"type": "linear", "size": 512},
            {"type": "batch_norm", "momentum": 0.2},
            {"type": "adaptive_dropout", "max_rate": 0.4},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "dropout", "rate": 0.2},
            {"type": "relu"}
        ]
    }

    results = []
    for name, layers in configs.items():
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=layers
        )
        
        history, time = train_model(model, train_loader, test_loader)
        stats = analyze_layers(model)
        
        results.append({
            "config": name,
            "train_acc": max(history["train_accs"]),
            "stats": stats
        })

    draw_adaptive_regularization(results, 'plots/adaptive.png')

if __name__ == '__main__':
    compare_regularization_technics()
    adaptive_regularization()
