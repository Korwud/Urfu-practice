import torch
import torch.nn as nn
from torchvision import transforms, models
from datasets import CustomImageDataset
from torch.utils.data import DataLoader
from custom_augs import RandomBlur, RandomPerspective, RandomBrightness, AugmentationPipeline
from extra_augs import ElasticTransform, AddGaussianNoise, AutoContrast
from utils import show_images, show_multiple_augmentations, show_single_augmentation, make_hystogramma, plot_bar_data, train_model, plot_training_history, save_model
import random
import os
from PIL import Image
import numpy as np
import psutil
import time


def basic_augmentations():
    """Стандартные аугментации torchvision"""

    transforms_list = [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomCrop(size=224, padding=16),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=1),
    ]

    transform_pipeline = transforms.Compose(transforms_list)
    dataset = CustomImageDataset('data/train')

    sample_images = []
    sample_labels = []
    selected_classes = set()

    for img, label in dataset:
        if label not in selected_classes and len(selected_classes) < 5:
            sample_images.append(img)
            sample_labels.append(label)
            selected_classes.add(label)
        if len(selected_classes) == 5:
            break

    for i, (original_img, label) in enumerate(zip(sample_images, sample_labels)):
        class_name = dataset.get_class_names()[label]

        augmented_images = []
        titles = []
        for transform in transforms_list:
            augmented_img = transform(original_img)
            augmented_images.append(augmented_img)
            titles.append(str(transform).split('(')[0])
        
        # Визуализируем отдельные аугментации
        show_multiple_augmentations(original_img, augmented_images, titles, f'results/exercise 1/{class_name}_multiple.png')
        
        # Применяем все аугментации вместе
        full_augmented = transform_pipeline(original_img)
        show_single_augmentation(original_img, full_augmented, f'results/exercise 1/{class_name}_single.png', "Все аугментации")
        

def custom_augmentations():
    """Кастомные аугментации"""

    extra_augs = [
        AddGaussianNoise(0., 0.2),
        ElasticTransform(1, 1, 50),
        AutoContrast(1)
    ]

    custom_augs = [
        RandomBlur(),
        RandomPerspective(),
        RandomBrightness()
    ]

    dataset = CustomImageDataset(root_dir='data/train')

    sample_images = []
    for i in range(3):
        random_number = random.randint(0, len(dataset))
        sample_images.append(dataset[random_number][0])

    titles = ['Extra', 'Custom']
    custom_titles = ['RandomBlur', 'RandomPerspective', 'RandomBrightness']
    extra_titles = ['GaussianNoise', 'ElasticTransform', 'AutoContrast']

    for i, original_img in enumerate(sample_images):
        custom_augmented = []
        for transform in custom_augs:
            custom_augmented.append(transform(original_img))
        
        show_multiple_augmentations(original_img, custom_augmented, custom_titles, f'results/exercise 2/{i+1}_custom.png')

        extra_augmented = []
        for transform in extra_augs:
            transform_aug = transforms.Compose([transforms.ToTensor(), transform])
            extra_augmented.append(transform_aug(original_img))
        
        show_multiple_augmentations(original_img, extra_augmented, extra_titles, f'results/exercise 2/{i+1}_extra.png')

        show_images(custom_augmented + extra_augmented, f'results/exercise 2/{i+1}_comparison.png', nrow=6, title="Сравнение аугментаций")


def analyze_dataset(root_dir='data/train'):
    """Анализ фотографий в датасете"""

    class_counts = {}
    size_stats = []
    
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                 
                img_path = os.path.join(class_dir, img_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                    size_stats.append((width, height))
    
    widths = [w for w, h in size_stats]
    heights = [h for w, h in size_stats]
    
    size_info = {
        'min_width': min(widths),
        'max_width': max(widths),
        'mean_width': np.mean(widths),
        'min_height': min(heights),
        'max_height': max(heights),
        'mean_height': np.mean(heights),
        'total_images': len(size_stats)
    }
    
    make_hystogramma(class_counts, widths, heights, size_stats, 'results/exercise 3/hystogramma.png')


def augmentation_pipeline():

    # Light
    light_aug = AugmentationPipeline()
    light_aug.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1))
    light_aug.add_augmentation('ColorJitter', transforms.ColorJitter(brightness=0.2, contrast=0.2))

    # Medium
    medium_aug = AugmentationPipeline()
    medium_aug.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1))
    medium_aug.add_augmentation('ColorJitter', transforms.ColorJitter(brightness=0.4, contrast=0.4))
    medium_aug.add_augmentation('RandomRotation', transforms.RandomRotation(15))

    # Heavy
    heavy_aug = AugmentationPipeline()
    heavy_aug.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=1))
    heavy_aug.add_augmentation('ColorJitter', transforms.ColorJitter(brightness=0.6, contrast=0.6))
    heavy_aug.add_augmentation('RandomRotation', transforms.RandomRotation(30))
    heavy_aug.add_augmentation('RandomResizedCrop', transforms.RandomResizedCrop(128, scale=(0.8, 1.0)))

    dataset = CustomImageDataset(root_dir='data/train')
    sample_images = []
    for i in range(3):
        random_number = random.randint(0, len(dataset))
        sample_images.append(dataset[random_number][0])

    light_aug_image = light_aug.apply(sample_images[0])
    medium_aug_image = medium_aug.apply(sample_images[1])
    heavy_aug_image = heavy_aug.apply(sample_images[2])

    show_single_augmentation(sample_images[0], light_aug_image, 'results/exercise 4/light_aug.png')
    show_single_augmentation(sample_images[1], medium_aug_image, 'results/exercise 4/medium_aug.png')
    show_single_augmentation(sample_images[2], heavy_aug_image, 'results/exercise 4/heavy_aug.png')


def run_size_experiment():
    sizes = [64, 128, 224, 512] 
    num_images = 100  
    results = {
        'sizes': [],
        'avg_load_times': [],
        'avg_aug_times': [],
        'memory_usage': []
    }

    augmentation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
    ])

    for size in sizes:
        dataset = CustomImageDataset(
            root_dir='data/train',
            target_size=(size, size)
        )
        
        mem_before = psutil.virtual_memory().used / (1024 ** 2)
        
        load_times = []
        for i in range(num_images):
            start_time = time.time()
            img, _ = dataset[i]
            load_times.append(time.time() - start_time)
        
        aug_times = []
        for i in range(num_images):
            img, _ = dataset[i]
            start_time = time.time()
            augmented_img = augmentation_pipeline(img)
            aug_times.append(time.time() - start_time)
        
        mem_after = psutil.virtual_memory().used / (1024 ** 2)
        mem_usage = mem_after - mem_before
        
        results['sizes'].append(size)
        results['avg_load_times'].append(sum(load_times) / num_images)
        results['avg_aug_times'].append(sum(aug_times) / num_images)
        results['memory_usage'].append(mem_usage)

    column_names = [str(size) for size in sizes]
    plot_bar_data(results['avg_load_times'], column_names, 'Сравнение среднего времени загрузки', 'results/exercise 5/avg_load_times.png')
    plot_bar_data(results['avg_aug_times'], column_names, 'Сравнение среднего времени аугментации', 'results/exercise 5/avg_aug_times.png')
    plot_bar_data(results['memory_usage'], column_names, 'Сравнение потребления памяти', 'results/exercise 5/memory_usage.png')


def pretraining_models():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset('data/train', transform=train_transform)
    val_dataset = CustomImageDataset('data/test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_classes = len(train_dataset.get_class_names())
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    train_history, val_history = train_model(model, train_loader, val_loader, train_dataset, val_dataset)
    print('Train history', train_history)
    print('Val history', val_history)
    plot_training_history(train_history, val_history, 'results/exercise 6/resnet_learning.png')
    save_model(model, 'results/exercise 6/model')



if __name__ == '__main__':
    '''basic_augmentations()
    custom_augmentations() 
    analyze_dataset()
    augmentation_pipeline()
    run_size_experiment()'''
    pretraining_models()
