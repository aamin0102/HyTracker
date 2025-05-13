import argparse
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from joblib import Parallel, delayed

from data_processing.dataset_helper import X2Cube2
from inference.band_selection_15 import band_selection_15
from inference.band_selection_16 import band_selection_16
from inference.band_selection_25 import band_selection_25
from inference.classifier import Classifier, set_optimizer, BCELoss
from inference.data_prov import RegionExtractor, crop_image2
from inference.evalution_metrices import calculate_euclidean_distance, check_bbox_changes
from inference.sample_generator import SampleGenerator, overlap_ratio
from utils.datasets import letterbox

from inference.new_helper import gen_config, X2Cube, calAUC, xyxy_to_xywh, \
    is_bbox_inside_image, gen_config2
from inference.sequence_output import SequenceModel
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = 32
    batch_neg = 96
    batch_test = 256
    batch_neg_cand = max(1024, batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def initialize_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                       default='weights\\best.pth', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.05, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    return opt


def process_image_vis(image_file):
    image = np.array(Image.open(image_file))
    processed_image = X2Cube(image)
    return processed_image

def process_image_rednir(image_file):
    image = np.array(Image.open(image_file))
    processed_image = X2Cube(image)
    processed_image = processed_image[:,:,0:15]
    return processed_image

def process_image_nir(image_file):
    image = np.array(Image.open(image_file))
    processed_image = X2Cube2(image)
    return processed_image

def create_pseudocolor_images(image, band_order):
    height, width, channel = image.shape
    images = []
    for i in range(int(channel//3)):
        im0 = np.zeros((height, width, 3))
        im0[:, :, 0] = image[:, :, band_order[i * 3 + 0]]
        im0[:, :, 1] = image[:, :, band_order[i * 3 + 1]]
        im0[:, :, 2] = image[:, :, band_order[i * 3 + 2]]
        images.append(im0)
    return images


def find_scores(classifier, image, bbox):
    regions = np.zeros((1, 107, 107, 3), dtype='uint8')
    crop_image = crop_image2(np.array(image), bbox, 107, 16)

    regions[0] = crop_image
    regions = regions.transpose(0, 3, 1, 2)
    regions = regions.astype('float32') - 128.
    regions = torch.from_numpy(regions)
    regions = regions.cuda()
    with torch.no_grad():
        feat = classifier(regions, out_layer='fc6')

    feats = feat.detach().clone()
    return feats[:, 1].cpu().detach().numpy()


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples)
    feats = []
    for i, regions in enumerate(extractor):
        regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def generate_bboxes_nearby(original_bbox, image_size, n=500, target_iou=0.7, iou_tolerance=0.05):
    """
    Generate `n` bounding boxes near the original bbox with an IoU around `target_iou`.

    Parameters:
    - original_bbox: List or array [x, y, w, h] representing the original bounding box.
    - image_size: Tuple (width, height) representing the image size.
    - n: Number of bounding boxes to generate.
    - target_iou: The target IoU value to match (default 0.7).
    - iou_tolerance: How much the IoU can deviate from the target IoU.

    Returns:
    - List of generated bounding boxes near the original bbox with IoU ~ target_iou.
    """
    generated_bboxes = []
    image_w, image_h = image_size
    original_bbox = np.array(original_bbox)

    while len(generated_bboxes) < n:
        # Generate random perturbations to the original bbox
        new_bbox = original_bbox.copy()

        # Random translation (shift)
        new_bbox[0] += np.random.uniform(-0.1, 0.1) * original_bbox[2]  # Shift x
        new_bbox[1] += np.random.uniform(-0.1, 0.1) * original_bbox[3]  # Shift y

        # Random scale (size change)
        new_bbox[2] *= np.random.uniform(0.9, 1.1)  # Scale width
        new_bbox[3] *= np.random.uniform(0.9, 1.1)  # Scale height

        # Ensure the bbox stays within image bounds
        new_bbox[0] = np.clip(new_bbox[0], 0, image_w - new_bbox[2])
        new_bbox[1] = np.clip(new_bbox[1], 0, image_h - new_bbox[3])
        new_bbox[2] = np.clip(new_bbox[2], 1, image_w - new_bbox[0])  # Ensure width is positive
        new_bbox[3] = np.clip(new_bbox[3], 1, image_h - new_bbox[1])  # Ensure height is positive

        # Calculate IoU with the original bounding box
        iou = overlap_ratio(new_bbox, original_bbox)

        # Check if the IoU is within the desired range
        if target_iou - iou_tolerance <= iou <= target_iou + iou_tolerance:
            generated_bboxes.append(new_bbox)

    return np.array(generated_bboxes)

def tracking_the_videos(frames, bboxes, class_name):
    resultArr[0] = gt_list[0]
    height, width, channel = psudo_images[0][0].shape
    # ===========================Tracking Implementation========================================
    target_bbox = gt_list[0]
    tracker_weight_path = "weights/mdnet_vot-otb.pth"
    classifier = Classifier(tracker_weight_path).cuda()
    criterion = BCELoss()
    classifier.set_learnable_params(["fc"])
    init_optimizer = set_optimizer(classifier, 0.0005, {'fc6': 10})
    pos_examples = SampleGenerator('gaussian', (width, height), 0.1, 1.3)(
        target_bbox, 500, [0.7, 1])
    if len(pos_examples) == 0:
        pos_examples = generate_bboxes_nearby(target_bbox, (width, height))
    neg_examples = np.concatenate([
        SampleGenerator('uniform', (width, height), 1, 1.6)(
            target_bbox, int(5000 * 0.5), [0, 0.5]),
        SampleGenerator('whole', (width, height))(
            target_bbox, int(5000 * 0.5), [0, 0.5])])
    neg_examples = np.random.permutation(neg_examples)
    pos_feats = forward_samples(classifier, psudo_images[0][0], pos_examples)
    neg_feats = forward_samples(classifier, psudo_images[0][0], neg_examples)
    train(classifier, criterion, init_optimizer, pos_feats, neg_feats, 50)
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()
    sample_generator = SampleGenerator('gaussian', (width, height), 0.6, 1.05)
    pos_generator = SampleGenerator('gaussian', (width, height), 0.1, 1.3)
    neg_generator = SampleGenerator('uniform', (width, height), 1, 1.6)
    neg_examples = neg_generator(target_bbox, 200, [0, 0.5])
    neg_feats = forward_samples(classifier, psudo_images[0][0], neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]
    update_optimizer = set_optimizer(classifier, 0.001, {'fc6': 10})
    threshold = 25


    for j in range(1, len(frames)):
        current_image = []
        best_conf = 0
        best_xyxy = [0, 0, 0, 0]

        # ================================tracker output==================================
        image = psudo_images[j][0]
        samples = sample_generator(target_bbox, 256)
        sample_scores = forward_samples(classifier, image, samples, out_layer='fc6').cpu()
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        # ===================================================================================

        # fusion of results
        current_bbox = [0, 0, 0, 0]
        current_best_conf = 0
        all_output_from_yolo = []
        other_bboxes = []

        for image_no in range(3):
            im0 = psudo_images[j][image_no]
            if image_no == 0:
                current_image = im0
            img = letterbox(np.copy(im0), 640, 32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    best_box = []
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'

                        a = torch.stack(xyxy).cpu().detach().numpy()
                        b = xyxy_to_xywh(a)
                        if cls == class_name and calculate_euclidean_distance(b, resultArr[j - 1]) < threshold:
                            all_output_from_yolo.append(b)

                        if best_conf < conf and cls == class_name and calculate_euclidean_distance(b, resultArr[
                            j - 1]) < threshold:
                            best_conf = conf
                            best_xyxy = b

                        overlap = overlap_ratio(np.array(target_bbox), np.array(b))
                        if class_name == cls and current_best_conf < conf and overlap > 0:
                            current_best_conf = conf
                            current_bbox = b
                            current_image = im0

        success = False
        if current_best_conf > 0.3:
            target_bbox = np.array(current_bbox)
            success = True
        # elif  best_conf>0.5:
        #    target_bbox = np.array(best_xyxy)
        else:
            target_score = target_score.cpu()
            # target_score = 9999
            for k in range(len(all_output_from_yolo)):
                score = find_scores(classifier, image, all_output_from_yolo[k])[0]
                if score > target_score:
                    target_score = score
                    target_bbox = np.array(all_output_from_yolo[k])

        if not is_bbox_inside_image(target_bbox, width, height) and sum(best_xyxy)!=0:
            target_bbox = best_xyxy

        elif check_bbox_changes(target_bbox, resultArr[j - 1]) and is_bbox_inside_image(target_bbox, width, height) and j>6:
            check_bboxes = resultArr[j - 6:j]
            target_bbox = sequence_model.get_output(check_bboxes)

        resultArr[j] = target_bbox

        copy_tracker = np.copy(target_bbox)
        copy_tracker[2] = copy_tracker[2] + copy_tracker[0]
        copy_tracker[3] = copy_tracker[3] + copy_tracker[1]
        label = f'{video_name}'
        plot_one_box(copy_tracker, current_image, label=label, color=colors[int(class_name + 1)%20], line_thickness=1)
        cv2.imshow(video_name, current_image / current_image.max())

        if success:
            threshold = 25
        else:
            threshold = threshold * 1.05

            # =============================================tracker update
            # Expand search area at failure
        if success == False:
            success = target_score > 0

        if success:
            sample_generator.set_trans(0.6)
        else:
            sample_generator.expand_trans(1.5)

        # Data collect
        if success:
            pos_examples = pos_generator(np.array(target_bbox), 50, [0.7, 1])
            if len(pos_examples) == 50:

                pos_feats = forward_samples(classifier, image, pos_examples)
                # print(pos_feats)
                pos_feats_all.append(pos_feats)
                if len(pos_feats_all) > 100:
                    del pos_feats_all[0]

                neg_examples = neg_generator(np.array(target_bbox), 200, [0, 0.3])
                neg_feats = forward_samples(classifier, image, neg_examples)
                neg_feats_all.append(neg_feats)
                if len(neg_feats_all) > 30:
                    del neg_feats_all[0]

        # Long term update
        if i % 10 == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(classifier, criterion, update_optimizer, pos_data, neg_data, 15)

    cv2.destroyAllWindows()
    return resultArr


if __name__=="__main__":
    # file_path = "Class_Representation/class_name.json"
    # with open(file_path, "r") as json_file:
    #     class_dictionary = json.load(json_file)
    sequence_model = SequenceModel()
    opt = initialize_all()
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    names.append("20")
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    file_path = "../class_representation/class_name_2023.json"
    with open(file_path, "r") as json_file:
        class_dictionary = json.load(json_file)

    dir = "D:\\hsi_2023_dataset\\validation\\hsi"
    # dir = "C:\\dataset\\hot_2024_dataset\\ranking\\hsi"

    video_types = os.listdir(dir)
    video_types.sort()
    all_results, all_gts = [], []

    # result_path = "../results/hot_2024_ranking"
    # os.makedirs(result_path)
    all_video_list = []
    total_auc = 0
    count = 1
    for _types in video_types:

        video_dir = os.path.join(dir, _types)
        video_list = os.listdir(video_dir)
        video_list.sort(reverse=True)
        for video_name in video_list:
            all_video_list.append(video_name)
            class_name = class_dictionary["validation_"+f"{_types}_" + video_name]
            print(f"video name: {video_name}", end=" ")
            img_list, gt_list = gen_config2(video_dir, video_name, image_format="HSI")
            if _types == "nir":
                band_order = band_selection_25(np.array(Image.open(img_list[0])), np.array(gt_list[0]))
                num_cores = 1  # Use all available CPU cores
                processed_images = Parallel(n_jobs=num_cores)(
                    delayed(process_image_nir)(image_file) for image_file in img_list)
                psudo_images = Parallel(n_jobs=num_cores)(
                    delayed(create_pseudocolor_images)(image, band_order[0]) for image in processed_images)
                # file = open(os.path.join(result_path, "nir_" + video_name + ".txt"), "x")
            elif _types == "rednir":
                # class_name = class_dictionary["val_rednir_" + video_name]
                band_order = band_selection_15(np.array(Image.open(img_list[0])), np.array(gt_list[0]))
                num_cores = 1  # Use all available CPU cores
                processed_images = Parallel(n_jobs=num_cores)(
                    delayed(process_image_rednir)(image_file) for image_file in img_list)
                psudo_images = Parallel(n_jobs=num_cores)(
                    delayed(create_pseudocolor_images)(image, band_order[0]) for image in processed_images)
                # file = open(os.path.join(result_path, "rednir_" + video_name + ".txt"), "x")
            elif _types == "vis":
                band_order = band_selection_16(np.array(Image.open(img_list[0])), np.array(gt_list[0]))
                num_cores = 1  # Use all available CPU cores
                processed_images = Parallel(n_jobs=num_cores)(
                    delayed(process_image_vis)(image_file) for image_file in img_list)
                psudo_images = Parallel(n_jobs=num_cores)(
                    delayed(create_pseudocolor_images)(image, band_order[0]) for image in processed_images)
                # file = open(os.path.join(result_path, "vis_" + video_name + ".txt"), "x")

            resultArr = np.zeros((len(img_list), 4))
            resultArr = tracking_the_videos(img_list, gt_list, class_name)
            all_results.append(resultArr)
            all_gts.append(gt_list)
            auc = calAUC([gt_list], [resultArr], [video_name])
            total_auc = total_auc + auc
            print(f"auc: {auc}, average: {total_auc / count}", )
            count += 1
            # for bb in resultArr:
            #     file.write("\t".join(map(str, bb)) + "\n")
            #
            # file.close()

    auc = calAUC(all_gts, all_results, all_video_list)
    print("Total auc:", auc)














