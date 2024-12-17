import os
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def X2Cube(img, B=[4, 4], skip = [4, 4],bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//B[0], N//B[1],bandNumber )
    #DataCube = DataCube.transpose(1, 0, 2)
    #DataCube = DataCube / DataCube.max() * 255
    #DataCube.astype('uint8')
    return DataCube

class ReconDataset(Dataset):
    def __init__(self, path):
        video_list = os.listdir(path)
        self.image_list = []
        self.gts = []
        for video in video_list:
            self.gen_config(path, video)
        self.index = list(range(0, len(self.image_list)))
        random.shuffle(self.index)

    def gen_config(self, path, video):
        video_path = os.path.join(path, video)
        images = [x for x in os.listdir(video_path) if x.find('png') != -1]
        images.sort()
        images = [video_path + '/' + x for x in images]

        gt_path = os.path.join(video_path, "groundtruth_rect.txt")
        gt_file = open(gt_path, "r")
        gts = gt_file.readlines()
        gts = [gt.split() for gt in gts]
        gts = [[float(x) for x in gt] for gt in gts]
        assert len(images)==len(gts), video +" images and gts are not matches"
        self.image_list = self.image_list + images
        self.gts = self.gts + gts

    def crop_image(self, image, bbox):
        target_height = target_width = 896
        x, y, w, h = bbox
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the half-width and half-height of the target size
        half_target_width = target_width // 2
        half_target_height = target_height // 2

        # Calculate the cropping region
        start_x = max(center_x - half_target_width, 0)
        end_x = min(center_x + half_target_width, image.shape[1])
        start_y = max(center_y - half_target_height, 0)
        end_y = min(center_y + half_target_height, image.shape[0])

        # Crop the image
        cropped_image = image[int(start_y):int(end_y), int(start_x):int(end_x)]

        # Pad the cropped image if its size is less than the target size
        if cropped_image.shape[0] < target_height or cropped_image.shape[1] < target_width:
            pad_top = max((target_height - cropped_image.shape[0]) // 2, 0)
            pad_bottom = max(target_height - cropped_image.shape[0] - pad_top, 0)
            pad_left = max((target_width - cropped_image.shape[1]) // 2, 0)
            pad_right = max(target_width - cropped_image.shape[1] - pad_left, 0)
            cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=0)

        return cropped_image


    def __len__(self):
        return len(self.index)


    def __getitem__(self, item):
        item = self.index[item]
        image = np.array(Image.open(self.image_list[item]))
        gt = self.gts[item]
        temp_gt = [(x*4) for x in gt]
        cropped_image = self.crop_image(image, temp_gt)
        cropped_image = X2Cube(cropped_image)/cropped_image.max()
        cropped_image = cropped_image.transpose((2, 0, 1)).astype(np.float32)
        return cropped_image