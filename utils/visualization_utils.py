import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def draw_heatmap(results, save_path):
    df = pd.DataFrame(results)
    pivot_table = df.pivot(index="Scheme", columns="Widths", values="Train Acc")

    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title("Точность моделей с разными схемами ширины слоев")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def draw_weigths(results, save_path):
    acc_df = pd.DataFrame([(x["Config"], x["Train Acc"]) for x in results],
                     columns=["Method", "Accuracy"])
    sns.barplot(x="Method", y="Accuracy", data=acc_df)

    plt.figure(figsize=(12, 6))
    for res in results:
        sns.kdeplot(res["Weights"], label=res["Config"])
    plt.legend()
    plt.title("Weight Distributions")
    plt.savefig(save_path)
    plt.show()


def draw_learning_stability(results, save_path):
    for res in results:
        plt.plot(res["History"]["train_accs"], label=f"{res['Config']} (train)")
        plt.plot(res["History"]["test_accs"], label=f"{res['Config']} (test)")
    plt.legend()
    plt.title("Training Stability")
    plt.savefig(save_path)
    plt.show()


def draw_adaptive_regularization(results, save_path):
    acc_df = pd.DataFrame([(x["config"], x["train_acc"]) for x in results],
                        columns=["Method", "Accuracy"])
    sns.barplot(x="Method", y="Accuracy", data=acc_df)

    for res in results:
        plt.figure()
        sns.lineplot(data=res["stats"], x="layer", y="std", marker="o")
        plt.title(f"{res['config']} - Weight Std by Layer")
        plt.savefig(save_path)
        plt.show()


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


def plot_by_epoch(history, save_path):
    """Визуализирует точность модели по эпохам"""

    train_accs = history['train_accs']
    test_accs = history['test_accs']

    n_epochs = len(train_accs)
    width = 0.4
    x = np.arange(n_epochs)

    x_train = x - width/2
    x_test = x + width/2

    plt.bar(x_train, train_accs, width=width, label='train_acc')
    plt.bar(x_test, test_accs, width=width, label='test_acc')

    plt.xticks(x, [f'Epoch {i+1}' for i in x])

    min_y = min(min(train_accs), min(test_accs))
    max_y = max(max(train_accs), max(test_accs))
    plt.ylim(min_y - 0.05, max_y + 0.05)

    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy per Epoch')

    plt.legend()
    plt.savefig(save_path)
    plt.show()