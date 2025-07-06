import torch
from models.custom_layers import CustomCNN
from models.cnn_models import SimpleCNN, ResNetTest
from utils.training_utils import evaluate_model, get_mnist_loaders, count_parameters
from utils.visualization_utils import plot_training_history, compare_models, plot_bar_data


def custom_layers_realisation():
    train_loader, test_loader = get_mnist_loaders()
    models = {
        "SimpleCNN": SimpleCNN(),
        "CustomCNN": CustomCNN()
    }
    
    histories = []
    for name, model in models.items():
        result = evaluate_model(model, train_loader, test_loader, epochs=3)
        histories.append(result['history'])
        print(result['history'])
        plot_training_history(result['history'], f'plots/3.1/{name}_history.png')

    compare_models(histories[0], histories[1], 'plots/3.1/comparison.png')


def residual_block_experiments():
    train_loader, test_loader = get_mnist_loaders()
    models = {
        'Basic': ResNetTest('basic'),
        'Bottleneck': ResNetTest('bottleneck'),
        'Wide': ResNetTest('wide')
    }

    histories = []
    parameters = []
    names = ['Basic', 'Bottleneck', 'Wide']

    for name, model in models.items():
        result = evaluate_model(model, train_loader, test_loader, epochs=2)
        histories.append(result['history'])
        print(result['history'])
        plot_training_history(result['history'], f'plots/3.2/{name}_history.png')
        parameters.append(count_parameters(model))

    plot_bar_data(parameters, names, 'Сравнение количества параметров моделей', 'plots/3.2/parameters_comparison.png')
    compare_models(histories[0], histories[1], 'plots/3.2/1_2_comparison.png')
    compare_models(histories[0], histories[2], 'plots/3.2/1_3_comparison.png')
    compare_models(histories[1], histories[2], 'plots/3.2/2_3_comparison.png')


if __name__ == '__main__':
    custom_layers_realisation()
    residual_block_experiments()