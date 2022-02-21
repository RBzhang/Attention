import numpy as np
import numpy.random as random

from PIL import Image, ImageFilter

import skimage.color

class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img


class RandomHEStain(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, img):
        img_he = skimage.color.rgb2hed(img)
        img_he[:, :, 0] = img_he[:, :, 0] * random.normal(1.0, 0.02, 1)  # H
        img_he[:, :, 1] = img_he[:, :, 1] * random.normal(1.0, 0.02, 1)  # E
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        img = Image.fromarray(np.uint8(img_rgb*255.999), img.mode)
        return img


class RandomGaussianNoise(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(random.normal(0.0, 0.5, 1)))
        return img


class HistoNormalize(object):
    """Normalizes the given PIL.Image"""

    def __call__(self, img):
        img_arr = np.array(img)
        # img_norm = normalize(img_arr)
        img = Image.fromarray(img_arr, img.mode)
        return img

