import cv2
import numpy as np


class GaussianBlur(object):
    """Implements gaussian blur augmentation, as described in SimCLR paper by Google

    This class works with Pytorch data augmentation pipeline
    https://github.com/google-research/simclr"""

    def __init__(self, kernel_size: int, *, min_sigma: float = 0.1, max_sigma: float = 2.0, prob: float = 0.5):
        """

        Args:
            kernel_size: size of the blur kernel. Kernel size should be odd number

            min_sigma: minimum value of sigma parameter

            max_sigma: maximum value of sigma parameter

            prob: probability of the augmentation. Default: 0.5 (50%)
        """

        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._prob = prob

        self._kernel_size = kernel_size

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        sample = np.array(sample)

        # blur the image with a self.prob chance
        prob = np.random.random_sample()

        if prob < self._prob:
            sigma = (self._max_sigma - self._min_sigma) * np.random.random_sample() + self._min_sigma
            sample = cv2.GaussianBlur(sample, (self._kernel_size, self._kernel_size), sigma)

        return sample
