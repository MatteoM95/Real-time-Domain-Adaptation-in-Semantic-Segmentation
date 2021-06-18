import torch
import glob
import os
from scipy import ndimage
from torchvision import transforms
import torchvision.transforms.functional
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from utils import get_label_info_IDDA, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, \
                  one_hot_it_v11_dice_IDDA, one_hot_it_v11_IDDA, colorize_mask
import random


def augmentation(image, label, p=0.5):
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    if random.random() < p:
        image = np.flip(image, 1)
        label = np.flip(label, 1)
    return image, label


def augmentation_pixel(image, filename="", p=0.5):
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    if random.random() < p:
        image = ndimage.gaussian_filter1d(image, 1, axis=1, mode='reflect')

    # img = Image.fromarray(image.astype('uint8'), 'RGB')
    # img.save("augmented/" + filename + "_C.png")
    # print(f"FILENAME 2: {filename}")

    return image


# ######################################################################################################################
# 1) open idda rgb image at index i
# 2) open its respective label
# 3) resize idda rgb image with 1280x720 dimension applying BILINEAR interpolation
# 4) resize its respective label with the same dimension but applying NEAREST interpolation
# 5) Random Crop rgb and label in the same position with a size equal to the camvid samples (960x720)
# 6) Return the modified image and label that now have a dimension of 960x720
# ######################################################################################################################

# image_path = ../IDDA/rgb
# label_path = ../IDDA/labels

# noinspection DuplicatedCode,PyShadowingNames
class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, json_path, scale=(720, 1280), crop=(720, 960), loss='dice', mode='train'):
        super().__init__()
        self.mode = mode

        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.jpg')))
        self.image_list.sort()

        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()

        self.label_info = get_label_info_IDDA(json_path)

        # resize
        self.camvid_resize = scale

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # self.cropped = transforms.RandomCrop(crop, pad_if_needed=True)
        self.image_size = crop
        self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])

        # IDDA resize and crop image/labe to have the same dimension as camvid image/label
        # =====================================
        img = transforms.Resize(self.camvid_resize, Image.BILINEAR)(img)
        img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        label = transforms.Resize(self.camvid_resize, Image.NEAREST)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        scale = random.choice(self.scale)
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        # =====================================
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)

            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        img = np.array(img)
        label = np.array(label)

        if self.mode == 'train':
            # set a probability of 0.5
            # augment image and label
            img, label = augmentation(img, label)

            # augment pixel image
            img = augmentation_pixel(img)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]

            # before, shape(720, 960, 3)
            label = one_hot_it_v11_dice_IDDA(label, self.label_info).astype(np.uint8)
            # after, shape(12, 720, 960)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11_IDDA(label, self.label_info).astype(np.uint8)
            # label = label.astype(np.float32)
            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    data = IDDA('../../datasets/IDDA/rgb',
                '../../datasets/IDDA/labels',
                '../../datasets/IDDA/classes_info.json',
                crop=(720, 960),
                loss='crossentropy',
                mode='train')
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info_IDDA('../../datasets/IDDA/classes_info.json')
    for i, (img, label) in enumerate(data):
        print(label.size())

        image = transforms.ToPILImage()(img)
        image.save("prova/" + str(i) + "_rgb.png")
        # label = colorize_mask(np.asarray(np.argmax(label, axis=0), dtype=np.uint8))
        label = colorize_mask(np.asarray(label, dtype=np.uint8))
        label.save("prova/" + str(i) + "_label.png")
        # print(torch.max(label))
