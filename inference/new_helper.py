import os

import cv2
import numpy as np
import torch
from PIL import Image

def gen_config2(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if len(gt_data_per_image)!=4:
            gt_data_per_image = line.split('\t')
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def gen_config(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name, image_format)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, image_format, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def gen_config2(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    if not os.path.isfile(gt_path):
        gt_path = os.path.join(video_dir, video_name, 'init_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if len(gt_data_per_image)!=4:
            gt_data_per_image = line.split('\t')
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    # assert len(images_in_video) == len(gts)

    return images_in_video, gts

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
    return DataCube

def X2Cube2(img, B=[5, 5], skip = [5, 5],bandNumber=25):
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
    return DataCube


def crop_image(image, bbox):
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


def cal_iou(box1, box2):
    """

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou

def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

def calAUC(gtArr,resArr, video_dir):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(len(resArr)):
        result_boxes = resArr[idx]
        result_boxes_gt = gtArr[idx]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]
        iou = list(map(cal_iou, result_boxes, result_boxes_gt))
        success = cal_success(iou)
        auc = np.mean(success)
        success_all_video.append(success)
        # print ('video = ',video_dir[idx],' , auc = ',auc)
    return np.mean(success_all_video)


def is_bbox_inside_image(bbox, image_width, image_height):
    x, y, w, h = bbox
    # Calculate the bottom-right coordinates
    x2 = x + w
    y2 = y + h

    # Check if the bounding box is inside the image
    if x >= 0 and y >= 0 and x2 <= image_width and y2 <= image_height:
        return True
    else:
        return False


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1  # Calculate width
    h = y2 - y1  # Calculate height
    return [x1, y1, w, h]


def is_bbox_inside_image(bbox, image_width, image_height):
    x, y, w, h = bbox
    # Calculate the bottom-right coordinates
    x2 = x + w
    y2 = y + h

    # Check if the bounding box is inside the image
    if x >= 0 and y >= 0 and x2 <= image_width and y2 <= image_height:
        return True
    else:
        return False




