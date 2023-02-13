#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import numbers
class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq_blur, seq_clear):
        for t in self.transforms:
            seq_blur, seq_clear = t(seq_blur, seq_clear)
        return seq_blur, seq_clear


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [Image.fromarray(np.uint8(img)) for img in seq_blur]
        seq_clear = [Image.fromarray(np.uint8(img)) for img in seq_clear]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            seq_blur  = [F.adjust_brightness(img, brightness_factor) for img in seq_blur]
            seq_clear = [F.adjust_brightness(img, brightness_factor) for img in seq_clear]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            seq_blur  = [F.adjust_contrast(img, contrast_factor) for img in seq_blur]
            seq_clear = [F.adjust_contrast(img, contrast_factor) for img in seq_clear]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            seq_blur  = [F.adjust_saturation(img, saturation_factor) for img in seq_blur]
            seq_clear = [F.adjust_saturation(img, saturation_factor) for img in seq_clear]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            seq_blur  = [F.adjust_hue(img, hue_factor) for img in seq_blur]
            seq_clear = [F.adjust_hue(img, hue_factor) for img in seq_clear]

        seq_blur  = [np.asarray(img) for img in seq_blur]
        seq_clear = [np.asarray(img) for img in seq_clear]

        seq_blur  = [img.clip(0,255) for img in seq_blur]
        seq_clear = [img.clip(0,255) for img in seq_clear]

        return seq_blur, seq_clear

class RandomColorChannel(object):
    def __call__(self, seq_blur, seq_clear):
        random_order = np.random.permutation(3)

        seq_blur  = [img[:,:,random_order] for img in seq_blur]
        seq_clear = [img[:,:,random_order] for img in seq_clear]

        return seq_blur, seq_clear

class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, seq_blur, seq_clear):

        shape = seq_blur[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        seq_blur = [img + gaussian_noise for img in seq_blur]
        seq_blur = [img.clip(0, 1) for img in seq_blur]

        return seq_blur, seq_clear

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [img/self.std -self.mean for img in seq_blur]
        seq_clear = [img/self.std -self.mean for img in seq_clear]

        return seq_blur, seq_clear

class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""

        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        seq_blur  = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear

class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        seq_blur  = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            seq_blur  = [np.copy(np.fliplr(img)) for img in seq_blur]
            seq_clear = [np.copy(np.fliplr(img)) for img in seq_clear]

        return seq_blur, seq_clear


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            seq_blur  = [np.copy(np.flipud(img)) for img in seq_blur]
            seq_clear = [np.copy(np.flipud(img)) for img in seq_clear]

        return seq_blur, seq_clear


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, seq_blur, seq_clear):
        seq_blur  = [np.transpose(img, (2, 0, 1)) for img in seq_blur]
        seq_clear = [np.transpose(img, (2, 0, 1)) for img in seq_clear]
        # handle numpy array
        seq_blur_tensor  = [torch.from_numpy(img).float() for img in seq_blur]
        seq_clear_tensor = [torch.from_numpy(img).float() for img in seq_clear]

        return seq_blur_tensor, seq_clear_tensor