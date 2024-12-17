import os

import cv2
import matplotlib.pyplot as plt
import torch

from HASBS.has_bs.has_bs import HAS_BS
from inference.new_helper import crop_image, X2Cube, X2Cube2


msbabs_25 = HAS_BS(in_channels=25)
pretrained_ = torch.load("../weights/band25_checkpoint.pth")
msbabs_25.load_state_dict(pretrained_)


def crop_image2(image, bbox):
    target_height = target_width = 224 * 5
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

def band_selection_25(image, bbox):
    gt = [int(x*5) for x in bbox]
    # gt = [int(x*4) for x in bbox]
    #crop_img = crop_image(image, gt)
    crop_img = crop_image2(image, gt)
    crop_img = X2Cube2(crop_img)
    # crop_img = crop_img[:,:,0:15]
    crop_img = crop_img.transpose(2, 0, 1)
    crop_img = crop_img / crop_img.max()
    crop_img = torch.from_numpy(crop_img)
    crop_img = crop_img.unsqueeze(0)
    crop_img, band_order = msbabs_25(crop_img.float())
    return band_order

