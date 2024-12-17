import numpy as np
import math

def calculate_bbox_area(bbox):
    _, _, w, h = bbox
    return w * h

def calculate_bbox_perimeter(bbox):
    _, _, w, h = bbox
    return 2 * (w + h)

def calculate_euclidean_distance(bbox1, bbox2):

    x1_center = bbox1[0] + bbox1[2] / 2
    y1_center = bbox1[1] + bbox1[3] / 2
    x2_center = bbox2[0] + bbox2[2] / 2
    y2_center = bbox2[1] + bbox2[3] / 2

    return math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def calculate_box_size_change(bbox1):
    x_1, y_1, w, h = bbox1
    x_2 = x_1 + w
    y_2 = y_1 + h
    return math.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)

def check_bbox_changes(bbox1, bbox2, area_threshold=0.2, perimeter_threshold=0.4, distance_threshold=0.4):
    """
    Check if the bounding box changes exceed the specified thresholds.

    Args:
        bbox1 (tuple): First bounding box represented as (x, y, w, h).
        bbox2 (tuple): Second bounding box represented as (x, y, w, h).
        area_threshold (float): Threshold for bounding box area change.
        perimeter_threshold (float): Threshold for bounding box perimeter change.
        distance_threshold (float): Threshold for Euclidean distance change.

    Returns:
        tuple: A tuple of Boolean values indicating whether the changes exceed the thresholds
               in the order (area_changed, perimeter_changed, distance_changed).
    """
    area_change = abs(calculate_bbox_area(bbox2) / calculate_bbox_area(bbox1) - 1)
    perimeter_change = abs(calculate_bbox_perimeter(bbox2) / calculate_bbox_perimeter(bbox1) - 1)
    euclidean_distance_change = abs(calculate_box_size_change(bbox1) / calculate_box_size_change(bbox2) - 1)

    area_changed = area_change > area_threshold
    perimeter_changed = perimeter_change > perimeter_threshold
    distance_changed = euclidean_distance_change > distance_threshold

    if distance_changed or perimeter_changed:
        if distance_changed:
            return True
        else:
            return False
    return False
    # return area_changed, perimeter_changed, distance_changed


def cal_iou(box1, box2):
    r"""

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
        print ('video = ',video_dir[idx],' , auc = ',auc)
    print('np.mean(success_all_video) = ', np.mean(success_all_video))


def calAUC_2(gtArr,resArr, video_dir):
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
        print ('video = ',video_dir[idx],' , auc = ',auc)