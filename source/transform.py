import cv2
import torch
import skimage
import numpy as np
from PIL import Image
from numpy import random
import torchvision.transforms as transforms
from scipy.ndimage import map_coordinates, gaussian_filter


class Compose(object):
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, img, mask, HE, edge) -> tuple[torch.Tensor | torch.FloatTensor, ...]:
        for t in self.transforms:
            img, mask, HE, edge = t(img, mask, HE, edge)

        return ConvertImgFloat(img, mask, HE, edge)


class ColorJitter(object):
    def __call__(self, img_path, mask_path):
        if random.randint(2):
            return skimage.io.imread(img_path), skimage.io.imread(mask_path)

        img = Image.open(img_path)
        img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)(img)
        return np.array(img), skimage.io.imread(mask_path)


def ConvertImgFloat(img, mask, HE, edge) -> tuple[torch.Tensor | torch.FloatTensor, ...]:
    if HE is None or HE.ndim != 3 or HE.shape[2] != 3:
        raise ValueError(f"Invalid HE shape: {HE.shape}")  # (64, 64, 3, 3)!

    img = torch.FloatTensor(np.transpose(np.maximum(img, 0), (2, 0, 1))/255.)
    HE = torch.FloatTensor(np.transpose(np.maximum(HE, 0), (2, 0, 1))/255.)
    mask = torch.FloatTensor(np.maximum(mask, 0))
    edge = torch.FloatTensor(np.maximum(edge, 0))
    if mask.max() > 1:
        mask = mask/255.
    if edge.max() > 1:
        edge = edge/255.

    return img, HE, mask, edge


class elastic_transform(object):
    def __call__(self, image, mask, HE, edge, alpha=120) -> tuple:
        if random.randint(2):
            return image, mask, HE, edge
        random_state = np.random.RandomState(None)
        shape3d = image.shape
        shape2d = mask.shape
        sigma = np.random.randint(8, 11)
        dx = gaussian_filter(
            (random_state.rand(*shape2d) * 2 - 1), sigma) * alpha
        dy = gaussian_filter(
            (random_state.rand(*shape2d) * 2 - 1), sigma) * alpha

        x3d, y3d, z = np.meshgrid(np.arange(shape3d[1]), np.arange(
            shape3d[0]), np.arange(shape3d[2]))
        x2d, y2d = np.meshgrid(np.arange(shape2d[1]), np.arange(shape2d[0]))
        y = np.ones(shape3d)
        x = np.ones(shape3d)
        y[:, :, 0] = dy
        y[:, :, 1] = dy
        y[:, :, 2] = dy
        x[:, :, 0] = dx
        x[:, :, 1] = dx
        x[:, :, 2] = dx
        indices2d = np.vstack(
            (np.reshape(y2d + dy, -1), np.reshape(x2d + dx, -1)))  # (2, N)
        indices3d = np.vstack(
            (np.reshape(y3d + y, -1), np.reshape(x3d + x, -1), np.reshape(z, -1)))  # (3, N)

        return map_coordinates(image, indices3d, order=1, mode='reflect').reshape(shape3d), \
            map_coordinates(mask, indices2d, order=1, mode='reflect').reshape(shape2d), \
            map_coordinates(HE, indices3d, order=1, mode='reflect').reshape(shape3d), \
            map_coordinates(edge, indices2d, order=1,
                            mode='reflect').reshape(shape2d)


class RandomSampleCrop(object):
    def __init__(self, min_win=0.4) -> None:
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            # (0.1, None),
            # (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

        self.min_win = min_win

    def __call__(self, img, mask, HE, edge) -> tuple[torch.FloatTensor, ...]:
        if random.randint(2):
            return img, mask, HE, edge

        height = np.shape(img)[0]
        width = np.shape(img)[1]
        while True:
            w = random.randint(int(self.min_win*width), width)
            h = random.randint(int(self.min_win*height), height)

            y1 = random.randint(0, h//2)
            x1 = random.randint(0, w//2)

            if y1+h >= height or x1+w >= width:
                continue
            rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])

            current_img = img[rect[0]:rect[2], rect[1]:rect[3], :]
            current_HE = HE[rect[0]:rect[2], rect[1]:rect[3], :]
            current_mask = mask[rect[0]:rect[2], rect[1]:rect[3]]
            current_edge = edge[rect[0]:rect[2], rect[1]:rect[3]]

            img = cv2.resize(current_img, (height, width),
                             interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(current_mask, (height, width),
                              interpolation=cv2.INTER_CUBIC)
            HE = cv2.resize(current_HE, (height, width),
                            interpolation=cv2.INTER_CUBIC)
            edge = cv2.resize(current_edge, (height, width),
                              interpolation=cv2.INTER_CUBIC)

            return img, mask, HE, edge


class RandomMirror_w(object):
    def __call__(self, img, mask, HE, edge) -> tuple[torch.FloatTensor, ...]:
        if random.randint(2):
            return img, mask, HE, edge

        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
        HE = HE[:, ::-1, :]
        edge = edge[:, ::-1]
        return img, mask, HE, edge


class RandomMirror_h(object):
    def __call__(self, img, mask, HE, edge) -> tuple[torch.FloatTensor, ...]:
        if random.randint(2):
            return img, mask, HE, edge

        img = img[::-1, :, :]
        mask = mask[::-1, :]
        HE = HE[::-1, :, :]
        edge = edge[::-1, :]
        return img, mask, HE, edge


class rotation(object):
    def __init__(self) -> None:
        self.angle = (90, 180, 270)

    def __call__(self, img, mask, HE, edge) -> tuple[cv2.typing.MatLike, ...]:
        if random.randint(2):
            return img, mask, HE, edge

        (h, w) = np.shape(img)[:2]
        (cx, cy) = (w//2, h//2)

        angle = random.choice(self.angle)

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
        HE = cv2.warpAffine(HE, M, (w, h))
        edge = cv2.warpAffine(edge, M, (w, h))
        return img, mask, HE, edge


class flip(object):
    def __init__(self) -> None:
        self.direction = (0, 1, -1)

    def __call__(self, img, mask, HE, edge) -> tuple[torch.FloatTensor, ...]:
        if random.randint(2):
            return img, mask, HE, edge

        direction = random.choice(self.direction)
        img = cv2.flip(img, direction)
        mask = cv2.flip(mask, direction)
        HE = cv2.flip(HE, direction)
        edge = cv2.flip(edge, direction)
        return img, mask, HE, edge
