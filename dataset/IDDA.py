import glob
import os
import random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from utils import get_label_info_IDDA, RandomCrop, one_hot_it_v11_dice_IDDA, one_hot_it_v11_IDDA, colorize_mask, \
    augmentation, augmentation_pixel


# noinspection PyShadowingNames
class IDDA(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, json_path, scale=(720, 1280), crop=(720, 960), loss='dice'):
        super().__init__()

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
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.image_size = crop
        # self.scale = [0.5, 1, 1.25, 1.5, 1.75, 2]
        self.loss = loss

    def __getitem__(self, index):
        # load image and crop
        seed = random.random()
        img = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])

        # image/label resize and crop to 1280x720
        # =====================================
        img = transforms.Resize(self.camvid_resize, Image.BILINEAR)(img)
        img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)
        label = transforms.Resize(self.camvid_resize, Image.NEAREST)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)
        # =====================================

        # scale = random.choice(self.scale)
        scale = 1
        scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

        # randomly resize image and random crop
        img = transforms.Resize(scale, Image.BILINEAR)(img)
        img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)

        label = transforms.Resize(scale, Image.NEAREST)(label)
        label = RandomCrop(self.image_size, seed, pad_if_needed=True)(label)

        img = np.array(img)
        label = np.array(label)

        # Augmentation
        img, label = augmentation(img, label)
        # img = augmentation_pixel(img)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        if self.loss == 'dice':
            # label -> [num_classes, H, W]

            # before shape(720, 960, 3)
            label = one_hot_it_v11_dice_IDDA(label, self.label_info).astype(np.uint8)
            # after shape(12, 720, 960)

            label = np.transpose(label, [2, 0, 1]).astype(np.float32)

            label = torch.from_numpy(label)

            return img, label

        elif self.loss == 'crossentropy':
            label = one_hot_it_v11_IDDA(label, self.label_info).astype(np.uint8)

            label = torch.from_numpy(label).long()

            return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    data = IDDA('../../datasets/IDDA/rgb',
                '../../datasets/IDDA/labels',
                '../../datasets/IDDA/classes_info.json',
                crop=(720, 960),
                loss='dice')

    # label_info = get_label_info_IDDA('../../datasets/IDDA/classes_info.json')
    for i, (img, label) in enumerate(data):
        print(label.size())

        image = transforms.ToPILImage()(img)
        image.save(str(i) + "_rgb.png")

        # # Dice loss version
        # =====================================
        # with open("prova_our_dice/" + str(i) + "_sm.txt", 'w') as outfile:
        #     outfile.write('# Array shape: {0}\n'.format(label.shape))
        #     for data_slice in label:
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         outfile.write('# New slice\n')
        #
        # label = colorize_mask(np.asarray(np.argmax(label, axis=0), dtype=np.uint8))
        # =====================================

        # # Crossentropy loss version
        # =====================================
        np.savetxt(str(i) + "_sm.txt", label, fmt='%-7.2f')
        label = colorize_mask(np.asarray(label, dtype=np.uint8))
        # =====================================

        label.save(str(i) + "_label.png")
        # print(torch.max(label))
