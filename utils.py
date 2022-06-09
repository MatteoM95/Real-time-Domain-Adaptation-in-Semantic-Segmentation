import numbers
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from scipy import ndimage

# color coding of semantic classes
palette = [119, 11, 32,  # 0 - bicycle
           70, 70, 70,  # 1 - building
           0, 0, 142,  # 2 - car
           153, 153, 153,  # 3 - pole
           190, 153, 153,  # 4 - fence
           220, 20, 60,  # 5 - Pedestrian
           128, 64, 128,  # 6 - Road
           244, 35, 232,  # 7 - Sidewalk
           220, 220, 0,  # 8 - SignSymbol
           70, 130, 180,  # 9 - Sky
           107, 142, 35,  # 10 - Tree
           0, 0, 0]  # 11 - Void
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9):
    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_11 = row['class_11']
        label[label_name] = [int(r), int(g), int(b), class_11]

    return label


def get_label_info_IDDA(json_path):
    ann = pd.read_json(json_path)
    label = []
    for iter, row in ann.iterrows():
        # label2camvid column is a list composed by [IDDA_class_index, CamVid_class_index].
        label.append(row["label2camvid"][1])

    return label


def one_hot_it_v11(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = np.zeros(label.shape[:-1])
    # from 0 to 11, and 11 means void
    class_index = 0
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)

            semantic_map[class_map] = class_index
            class_index += 1
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = 11
    return semantic_map


def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> shape(H, W, class_num)
    semantic_map = []
    void = np.zeros(label.shape[:2])
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)

            semantic_map.append(class_map)
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            void[class_map] = 1
    semantic_map.append(void)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    return semantic_map


def one_hot_it_v11_IDDA(label, label_info):
    semantic_map = np.zeros(label.shape[:2])

    for x, y in np.ndindex(label.shape[:2]):
        c = label[x][y][0]
        camvid_class = label_info[c]

        if camvid_class == 255:
            semantic_map[x][y] = 11
        else:
            semantic_map[x][y] = camvid_class

    return semantic_map


def one_hot_it_v11_dice_IDDA(label, label_info):
    shape = label.shape[:2] + (12,)
    semantic_map = np.zeros(shape)

    for x, y in np.ndindex(label.shape[:2]):
        c = label[x][y][0]
        camvid_class = label_info[c]
        if camvid_class == 255:
            semantic_map[x][y][11] = 1
        else:
            semantic_map[x][y][camvid_class] = 1

    return semantic_map


def reverse_one_hot(image):
    """
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def fast_hist(a, b, n):
    '''
	a and b are predict and mask respectively
	n is the number of classes
	'''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

    def __init__(self, size, seed, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.seed = seed

    @staticmethod
    def get_params(img, output_size, seed):
        """Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
        random.seed(seed)
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
        if self.padding > 0:
            img = torchvision.transforms.functional.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.seed)

        return torchvision.transforms.functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def cal_miou(miou_list, csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    miou_dict = {}
    cnt = 0
    for iter, row in ann.iterrows():
        label_name = row['name']
        class_11 = int(row['class_11'])
        if class_11 == 1:
            miou_dict[label_name] = miou_list[cnt]
            cnt += 1
    return miou_dict, np.mean(miou_list)


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
