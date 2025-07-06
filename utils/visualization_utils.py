import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def create_table(data, headers=None):
    """Создает и выводит таблицу из переданных данных"""
    table = tabulate(data, headers=headers, tablefmt='grid')
    print(table)


def plot_training_history(history, save_path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_bar_data(data, column_names, title, save_path):
    """Визуализирует переданные данные в виде столбчатой диаграммы"""
    x = range(len(data))

    plt.bar(x, data)
    plt.xticks([])

    for i, v in enumerate(data):
        plt.text(i, -1, column_names[i], ha='center', va='top')

    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(first_history, second_history, save_path):
    """Сравнивает результаты двух сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(first_history['test_accs'], label='First Network', marker='o')
    ax1.plot(second_history['test_accs'], label='Second Network', marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(first_history['test_losses'], label='First Network', marker='o')
    ax2.plot(second_history['test_losses'], label='Second Network', marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show() 


def plot_gradient_flow(gradients, model_name, save_path):
    plt.figure(figsize=(12, 6))
    for name, grad in gradients.items():
        if 'weight' in name: 
            plt.plot(grad, label=name)
    
    plt.title(f'Gradient Flow - {model_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Gradient Magnitude')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'plots/{model_name}_confusion.png')
    plt.show()


def plot_activations(activations, title, n=16):
    plt.figure(figsize=(12, 8))
    activations = activations.cpu().numpy()
    
    for i in range(min(n, activations.shape[1])):
        plt.subplot(4, 4, i+1)
        plt.imshow(activations[0, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i+1}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'plots/{title}_activations.png')
    plt.show()


def visualize_feature_maps_simple(model, test_loader, device, save_path, layer_names=None):
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:1].to(device)
    
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    hooks = []
    for name, layer in model.named_modules():
        if layer_names is None or name in layer_names:
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        model(data)
    
    for name, act in activations.items():
        plt.figure(figsize=(12, 2))
        plt.title(name)
        for i in range(min(8, act.size(1))):
            plt.subplot(1, 8, i+1)
            plt.imshow(act[0, i].cpu().numpy())
            plt.axis('off')
        plt.savefig(save_path)
        plt.show()
    
    for hook in hooks:
        hook.remove()