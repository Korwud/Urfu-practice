import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim



def show_images(images, save_path, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [
        resize_transform(Image.fromarray(img) if isinstance(img, np.ndarray) else img)
        for img in images
    ]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = np.array(img)
        # Нормализуем для отображения
        if isinstance(img, torch.Tensor):
            img_np = np.clip(img_np, 0, 1).transpose(1, 2, 0)
        else:
            img_np = np.clip(img_np / 255, 0, 1).transpose(0, 1, 2)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def show_single_augmentation(original_img, augmented_img, save_path, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Оригинальное изображение
    orig_np = np.array(orig_resized)
    orig_np = np.clip(orig_np / 255, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = np.array(aug_resized)
    aug_np = np.clip(aug_np / 255, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def show_multiple_augmentations(original_img, augmented_imgs, titles, save_path):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = np.array(orig_resized)
    orig_np = np.clip(orig_np / 255, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        if isinstance(aug_img, np.ndarray):
            aug_img = Image.fromarray(aug_img)
        aug_resized = resize_transform(aug_img)
        aug_np = np.array(aug_resized)
        if isinstance(aug_resized, torch.Tensor):
            aug_np = aug_np.transpose(1, 2, 0)
            aug_np = np.clip(aug_np, 0, 1)
        else:
            aug_np = np.clip(aug_np / 255, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show() 


def make_hystogramma(class_counts, widths, heights, size_stats, save_path):
    plt.figure(figsize=(15, 5))
    
    # Гистограмма распределения классов
    plt.subplot(1, 3, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Распределение изображений по классам')
    plt.xlabel('Классы')
    plt.ylabel('Количество изображений')
    plt.xticks(rotation=45, ha='right')
    
    # Распределение размеров (ширина)
    plt.subplot(1, 3, 2)
    plt.hist(widths, bins=30, alpha=0.7, label='Ширина')
    plt.hist(heights, bins=30, alpha=0.7, label='Высота')
    plt.title('Распределение размеров изображений')
    plt.xlabel('Пиксели')
    plt.ylabel('Количество')
    plt.legend()
    
    # Соотношение сторон
    plt.subplot(1, 3, 3)
    aspect_ratios = [w/h for w, h in size_stats]
    plt.hist(aspect_ratios, bins=30)
    plt.title('Распределение соотношений сторон')
    plt.xlabel('Ширина/Высота')
    plt.ylabel('Количество')
    
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


def plot_training_history(train_history, val_history, save_path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_history['losses'], label='Train Loss')
    ax1.plot(val_history['losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(train_history['accs'], label='Train Acc')
    ax2.plot(val_history['accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def train_model(model, train_loader, val_loader, train_dataset, val_dataset):
    train_history = {
        'losses': [],
        'accs': []
    }
    val_history = {
        'losses': [],
        'accs': []
    }

    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):      
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        train_history['losses'].append(epoch_loss)
        train_history['accs'].append(epoch_acc)
        
        # Валидация
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        
        val_history['losses'].append(val_epoch_loss)
        val_history['accs'].append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        print('-' * 10)

    return train_history, val_history

def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)