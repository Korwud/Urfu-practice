import torch
import random
import numpy as np
import cv2

class RandomBlur():
    def __init__(self, max_kernel_size=5):
        self.max_kernel_size = max_kernel_size

    def __call__(self, img):
        img = np.array(img)

        ksize = random.choice([k for k in range(1, self.max_kernel_size + 1) if k % 2 == 1])
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return blurred_img
    
class RandomPerspective:
    def __init__(self, max_shift=0.2):
        self.max_shift = max_shift 

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        # Генерируем случайные смещения для углов
        shift_x = self.max_shift * w
        shift_y = self.max_shift * h

        pts1 = np.float32([[0,0], [w,0], [w,h], [0,h]])
        pts2 = np.float32([
            [random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [w - random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [w - random.uniform(0, shift_x), h - random.uniform(0, shift_y)],
            [random.uniform(0, shift_x), h - random.uniform(0, shift_y)]
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_img = cv2.warpPerspective(img, matrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
        return warped_img
    
class RandomBrightness:
    def __init__(self, brightness_range=(0.8, 1.2)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        img = np.array(img)
        factor = random.uniform(*self.brightness_range)
        # Умножаем изображение на фактор яркости
        bright_img = np.clip(img * factor, 0, 255).astype(np.uint8)
        return bright_img
    
class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}
    
    def add_augmentation(self, name, aug):
        self.augmentations[name] = aug
    
    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]
    
    def apply(self, image):
        augmented_image = image
        for name, aug in self.augmentations.items():
            augmented_image = aug(augmented_image)
        return augmented_image
    
    def get_augmentations(self):
        return list(self.augmentations.keys())