import torch
from utils.training_utils import get_mnist_loaders, count_parameters, evaluate_model
from models.cnn_models import ConvConfig, ConvNetMNIST, get_cnn_models
from utils.visualization_utils import plot_activations, plot_training_history, save_model, plot_bar_data, plot_gradient_flow, visualize_feature_maps_simple, create_table


def kernel_research():
    train_loader, test_loader = get_mnist_loaders()

    configs = [
        ConvConfig([3], [32]),  
        ConvConfig([5], [16]), 
        ConvConfig([7], [8]),  
        ConvConfig([1, 3], [16, 16])
    ]

    model_names = ['3х3', '5х5', '7х7', '1х1+3х3']
    accuracies = []
    learning_times = []

    for i, config in enumerate(configs):
        print(f"\n=== Конфигурация {i+1}: Ядра {config.kernel_sizes} ===")
        model = ConvNetMNIST(config)
        print(f"Количество параметров: {count_parameters(model):,}")
        
        data = evaluate_model(model, train_loader, test_loader, epochs=3)
        
        accuracies.append(max(data['history']['test_accs']))
        learning_times.append(data['learning_time'])

        sample, _ = next(iter(train_loader))
        _ = model(sample[:1].to(next(model.parameters()).device))
        plot_activations(model.activations, f"mnist_kernel_{'_'.join(map(str, config.kernel_sizes))}")
        plot_training_history(data['history'], f'plots/2.1/kernel-{model_names[i]}')
        save_model(model, f'results/architecture_analysis/{model_names[i]}')
    create_table([accuracies], model_names)
    plot_bar_data(learning_times, model_names, 'Сравнение времени обучения', 'plots/2.1/kernel_learning_times.png')



def depth_cnn_research():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders()
    
    models = get_cnn_models()
    model_names = ['ShallowCNN', 'MediumCNN', 'DeepCNN', 'ResCNN']
    results = {}
    accuracies = []
    learning_times = []
    
    for name, model in models.items():
        model = model
        
        data = evaluate_model(model, train_loader, test_loader, epochs=3, device=device)
        results[name] = data

        plot_training_history(data['history'], f'plots/2.2/{name}_history.png')
        accuracies.append(max(data['history']['test_accs']))
        learning_times.append(data['learning_time'])

        print('Accuracies:', accuracies)
        print('Learning Times:', learning_times)
        
        plot_gradient_flow(data['gradients'], name, f'plots/2.2/{name}_gradient_flow.png')
        
        if name == "ResCNN":
            visualize_feature_maps_simple(model, test_loader, device, 'plots/2.2/feature_maps.png')
    
    create_table([accuracies], model_names)
    plot_bar_data(learning_times, model_names, 'Сравнение времени обучения', 'plots/2.2/depth_cnn_learning_times.png')

    return results


if __name__ == '__main__':
    kernel_research()
    depth_cnn_research()
