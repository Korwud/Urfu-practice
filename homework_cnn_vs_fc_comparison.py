import torch
from models.cnn_models import SimpleCNN, CNNWithResidual, RegularizedCNNWithResidual
from models.fc_models import FullyConnectedModel
from utils.training_utils import get_mnist_loaders, get_cifar_loaders, evaluate_model
from utils.visualization_utils import plot_training_history, plot_bar_data, save_model, plot_gradient_flow, plot_confusion_matrix, create_table


def compare_mnist_models():
    train_loader, test_loader = get_mnist_loaders()

    fc_model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"}
        ]
    )

    simple_cnn_model = SimpleCNN(input_channels=1, num_classes=10)

    residual_block_cnn_model = CNNWithResidual(input_channels=1, num_classes=10)

    fc_data = evaluate_model(fc_model, train_loader, test_loader)
    simple_cnn_data = evaluate_model(simple_cnn_model, train_loader, test_loader)
    residual_block_cnn_data = evaluate_model(residual_block_cnn_model, train_loader, test_loader)

    plot_training_history(fc_data['history'], 'plots/1.1/fc_learning.png')
    plot_training_history(simple_cnn_data['history'], 'plots/1.1/simple_cnn_learning.png')
    plot_training_history(residual_block_cnn_data['history'], 'plots/1.1/residual_block_cnn_learning.png')

    models = ['Fully connected', 'Simple CNN', 'Residual CNN']
    accuracies = [fc_data['accuracy'], simple_cnn_data['accuracy'], residual_block_cnn_data['accuracy']]
    learning_times = [fc_data['learning_time'], simple_cnn_data['learning_time'], residual_block_cnn_data['learning_time']]
    inference_times = [fc_data['inference_time'], simple_cnn_data['inference_time'], residual_block_cnn_data['inference_time']]
    parameters_count = [fc_data['parameters_count'], simple_cnn_data['parameters_count'], residual_block_cnn_data['parameters_count']]

    create_table([accuracies], models)
    plot_bar_data(learning_times, models, 'Сравнение времени обучения', 'plots/1.1/mnist_learning_times.png')
    plot_bar_data(inference_times, models, 'Сравнение inference времени', 'plots/1.1/mnist_inference_times.png')
    plot_bar_data(parameters_count, models, 'Сравнение количества параметров', 'plots/1.1/mnist_parameters.png')

    save_model(fc_model, 'results/mnist_comparison/fc_model')
    save_model(simple_cnn_model, 'results/mnist_comparison/simple_cnn_model')
    save_model(residual_block_cnn_model, 'results/mnist_comparison/residual_block_cnn_model')


def compare_cifar_models():
    train_loader, test_loader = get_cifar_loaders()
    cifar3_classes = ['class1', 'class2', 'class3']

    deep_fc_model = FullyConnectedModel(
        input_size=32*32*3,
        num_classes=10,
        layers=[
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    )

    residual_block_cnn_model = CNNWithResidual(input_channels=3, num_classes=10)

    regularized_residual_cnn_model = RegularizedCNNWithResidual()

    models = [deep_fc_model, residual_block_cnn_model, regularized_residual_cnn_model]
    models_names = ['Fully connected', 'Simple CNN', 'Residual CNN']
    accuracies = []
    learning_times = []

    for i in range(len(models)):
        data = evaluate_model(models[i], train_loader, test_loader)
        plot_training_history(data['history'], f'plots/1.2/cifar_model-{i + 1}_learning.png')
        plot_confusion_matrix(data['confusion_matrix'], cifar3_classes, models_names[i])
        plot_gradient_flow(data['gradients'], models_names[i], f'plots/{models_names[i]}_gradient_flow.png')
        accuracies.append(data['accuracy'])
        learning_times.append(data['learning_time'])
        save_model(models[i], f'results/cifar_comparison/{models_names[i]}')

    plot_bar_data(accuracies, models_names, 'Сравнение точности', 'plots/1.2/cifar_accuracies.png')
    plot_bar_data(learning_times, models_names, 'Сравнение времени обучения', 'plots/1.2/cifar_learning_times.png')


if __name__ == '__main__':
    compare_mnist_models()
    compare_cifar_models()
