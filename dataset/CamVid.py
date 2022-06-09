import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from utils import RandomCrop, one_hot_it_v11, one_hot_it_v11_dice, colorize_mask, augmentation, augmentation_pixel, \
                  get_label_info


# noinspection PyShadowingNames
class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, loss='dice', mode='train'):
        super().__init__()
        self.mode = mode

        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()

        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()

        self.label_info = get_label_info(csv_path)

        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.image_size = scale
        # self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])

        # scale = random.choice(self.scale)
        scale = 1
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # image and label resize and random crop
        if self.mode == 'train':
            img = transforms.Resize(scale, Image.BILINEAR)(img)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
            label = transforms.Resize(scale, Image.NEAREST)(label)
            label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

        img = np.array(img)
        label = np.array(label)

        # augment image and label
        if self.mode == 'train':
            # set a probability of 0.5
            img, label = augmentation(img, label)
            # img = augmentation_pixel(img)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]

            # before shape(3, 720, 960)
            label = one_hot_it_v11_dice(label, self.label_info).astype(np.uint8)
            # after shape(12, 720, 960)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)

            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11(label, self.label_info).astype(np.uint8)

            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640,
    # 640))
    data = CamVid(['../../datasets/CamVid/train', '../../datasets/CamVid/val'],
                  ['../../datasets/CamVid/train_labels', '../../datasets/CamVid/val_labels'],
                  '../../datasets/CamVid/class_dict.csv',
                  (720, 960), loss='dice', mode='train')
    from utils import get_label_info

    label_info = get_label_info('../../datasets/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(label.size())
        # print(torch.max(label))

        image = transforms.ToPILImage()(img)
        image.save("prova_camvid/" + str(i) + "_rgb.png")
        label = colorize_mask(np.asarray(np.argmax(label, axis=0), dtype=np.uint8))
        label.save("prova_camvid/" + str(i) + "_label.png")
