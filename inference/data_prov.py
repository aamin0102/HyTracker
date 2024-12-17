import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np


def crop_image2(img, bbox, img_size=107, padding=16, flip=False, rotate_limit=0, blur_limit=0):
    x, y, w, h = np.array(bbox, dtype='float32')

    cx, cy = x + w/2, y + h/2

    if padding > 0:
        w += 2 * padding * w/img_size
        h += 2 * padding * h/img_size

    # List of transformation matrices
    matrices = []

    # Translation matrix to move patch center to origin
    translation_matrix = np.asarray([[1, 0, -cx],
                                     [0, 1, -cy],
                                     [0, 0, 1]], dtype=np.float32)
    matrices.append(translation_matrix)

    # Scaling matrix according to image size
    scaling_matrix = np.asarray([[img_size / w, 0, 0],
                                 [0, img_size / h, 0],
                                 [0, 0, 1]], dtype=np.float32)
    matrices.append(scaling_matrix)

    # Define flip matrix
    if flip and np.random.binomial(1, 0.5):
        flip_matrix = np.eye(3, dtype=np.float32)
        flip_matrix[0, 0] = -1
        matrices.append(flip_matrix)

    # Define rotation matrix
    if rotate_limit and np.random.binomial(1, 0.5):
        angle = np.random.uniform(-rotate_limit, rotate_limit)
        alpha = np.cos(np.deg2rad(angle))
        beta = np.sin(np.deg2rad(angle))
        rotation_matrix = np.asarray([[alpha, -beta, 0],
                                      [beta, alpha, 0],
                                      [0, 0, 1]], dtype=np.float32)
        matrices.append(rotation_matrix)

    # Translation matrix to move patch center from origin
    revert_t_matrix = np.asarray([[1, 0, img_size / 2],
                                  [0, 1, img_size / 2],
                                  [0, 0, 1]], dtype=np.float32)
    matrices.append(revert_t_matrix)

    # Aggregate all transformation matrices
    matrix = np.eye(3)
    for m_ in matrices:
        matrix = np.matmul(m_, matrix)

    # Warp image, padded value is set to 128
    patch = cv2.warpPerspective(img,
                                matrix,
                                (img_size, img_size),
                                borderValue=128)

    if blur_limit and np.random.binomial(1, 0.5):
        blur_size = np.random.choice(np.arange(1, blur_limit + 1, 2))
        patch = cv2.GaussianBlur(patch, (blur_size, blur_size), 0)

    return patch


class RegionExtractor():
    def __init__(self, image, samples, flag=False, opts=None):
        self.image = np.asarray(image)
        self.samples = np.asarray(samples)

        self.crop_size = 107
        self.padding = 16
        self.batch_size = 256

        self.index = np.arange(len(samples))
        self.pointer = 0
        self.flag=flag

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
            if self.flag:
                plt.imshow((regions[i]))
                plt.show()
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


