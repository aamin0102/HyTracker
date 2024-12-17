import itertools
import json
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from matplotlib import patches

from band_selection_25 import band_selection
from dataset_helper import *


def image_storage(aug_image, copy_gt, training_image_count, path):
    if random.random() < 0.3:
        aug_image = augmented_image(aug_image)
    if random.random() < 0.3:
        aug_image, copy_gt = random_flip(aug_image, copy_gt)

    if random.random() < 0.3:
        aug_image, copy_gt = rotate_image_with_bounding_box(aug_image, copy_gt)

    image_path = os.path.join(path, 'nir_{:010d}.jpg'.format(training_image_count + 1))
    cv2.imwrite(image_path, aug_image)
    text_path = os.path.join(path, 'nir_{:010d}.txt'.format(training_image_count + 1))
    h, w, _ = aug_image.shape
    yolo_gt = convert_to_yolo(copy_gt, w, h)
    if os.path.isfile(text_path):
        with open(text_path, 'w') as file:
            file.write(str(number_of_class) + "\t")
            for bb in yolo_gt:
                file.write(str(bb) + "\t")
            file.write("\n")
    else:
        with open(text_path, 'x') as file:
            file.write(str(number_of_class) + "\t")
            for bb in yolo_gt:
                file.write(str(bb) + "\t")
            file.write("\n")


def validation_image_storage(aug_image, copy_gt, training_image_count, path):
    if random.random() < 0.7:
        aug_image, copy_gt = crop_and_paste(np.array(aug_image), copy_gt)
    if random.random() < 0.3:
        aug_image = augmented_image(aug_image)
    if random.random() < 0.3:
        aug_image, copy_gt = random_flip(aug_image, copy_gt)

    if random.random() < 0.3:
        aug_image, copy_gt = rotate_image_with_bounding_box(aug_image, copy_gt)

    image_path = os.path.join(path, 'nir_{:010d}.jpg'.format(training_image_count + 1))
    cv2.imwrite(image_path, aug_image)
    text_path = os.path.join(path, 'nir_{:010d}.txt'.format(training_image_count + 1))
    h, w, _ = aug_image.shape
    yolo_gt = convert_to_yolo(copy_gt, w, h)
    if os.path.isfile(text_path):
        with open(text_path, 'w') as file:
            file.write(str(number_of_class) + "\t")
            for bb in yolo_gt:
                file.write(str(bb) + "\t")
            file.write("\n")
    else:
        with open(text_path, 'x') as file:
            file.write(str(number_of_class) + "\t")
            for bb in yolo_gt:
                file.write(str(bb) + "\t")
            file.write("\n")


if __name__ == "__main__":

    training_dataset_directory = "D:\\hsi_2023_dataset\\training\\hsi\\nir"
    training_videos = os.listdir(training_dataset_directory)
    training_videos.sort()

    validation_dataset_directory = "D:\\hsi_2023_dataset\\validation\\hsi\\nir"
    validation_videos = os.listdir(validation_dataset_directory)
    validation_videos.sort()

    dataset_path = os.path.join("../", "hsi_dataset")

    training_path = os.path.join(dataset_path, "training")
    validation_path = os.path.join(dataset_path, "validation")

    training_image_count = 0
    validation_image_count = 0
    class_counter = 0

    file_path = "../class_representation/class_name_2023.json"
    with open(file_path, "r") as json_file:
        class_name = json.load(json_file)


    for i in range(len(training_videos)):
        hsi_img_list, hsi_gt_list = gen_config(training_dataset_directory, training_videos[i], image_format="HSI")
        index = np.array(range(len(hsi_img_list)))
        np.random.shuffle(index)
        no_of_image_in_training = int(len(hsi_img_list) * 0.80)

        training_set = index[0:no_of_image_in_training]
        validation_set = index[no_of_image_in_training:]
        number_of_class = class_name["training_"+"nir_"+training_videos[i]]
        print("Training Videos: ", training_videos[i], "Class: ", number_of_class)
        for j in range(len(training_set)):
            hsi_image = np.array(Image.open(hsi_img_list[training_set[j]]))
            hsi_cube = X2Cube2(hsi_image)
            hsi_gt = np.array(hsi_gt_list[training_set[j]])
            band_order = band_selection(hsi_image, hsi_gt)

            h, w, _ = hsi_cube.shape
            for k in range(int(len(band_order) / 3)):
                aug_image = np.zeros((h, w, 3))
                aug_image[:, :, 0] = hsi_cube[:, :, band_order[k * 3 + 0]]
                aug_image[:, :, 1] = hsi_cube[:, :, band_order[k * 3 + 1]]
                aug_image[:, :, 2] = hsi_cube[:, :, band_order[k * 3 + 2]]
                aug_image = aug_image / aug_image.max() * 255
                aug_image = np.uint8(aug_image)
                image_storage(aug_image, np.copy(hsi_gt), training_image_count, training_path)
                training_image_count += 1

        for j in range(len(validation_set)):
            hsi_image = np.array(Image.open(hsi_img_list[validation_set[j]]))
            hsi_cube = X2Cube2(hsi_image)
            hsi_gt = np.array(hsi_gt_list[validation_set[j]])
            band_order = band_selection(hsi_image, hsi_gt)
            h, w, _ = hsi_cube.shape
            for k in range(int(len(band_order) / 3)):
                aug_image = np.zeros((h, w, 3))
                aug_image[:, :, 0] = hsi_cube[:, :, band_order[k * 3 + 0]]
                aug_image[:, :, 1] = hsi_cube[:, :, band_order[k * 3 + 1]]
                aug_image[:, :, 2] = hsi_cube[:, :, band_order[k * 3 + 2]]
                aug_image = aug_image / aug_image.max() * 255
                aug_image = np.uint8(aug_image)
                image_storage(aug_image, np.copy(hsi_gt), validation_image_count, validation_path)
                validation_image_count += 1


    for i in range(len(validation_videos)):
        hsi_img_list, hsi_gt_list = gen_config(validation_dataset_directory, validation_videos[i], image_format="HSI")
        number_of_class = class_name["validation_"+"nir_"+validation_videos[i]]
        print("Validation Videos: ", validation_videos[i], "Class: ", number_of_class)
        for j in range(1000):
            hsi_image = np.array(Image.open(hsi_img_list[0]))
            hsi_cube = X2Cube2(hsi_image)
            hsi_gt = np.array(hsi_gt_list[0])
            band_order = band_selection(hsi_image, hsi_gt)

            h, w, _ = hsi_cube.shape
            for k in range(int(len(band_order) / 3)):
                aug_image = np.zeros((h, w, 3))
                aug_image[:, :, 0] = hsi_cube[:, :, band_order[k * 3 + 0]]
                aug_image[:, :, 1] = hsi_cube[:, :, band_order[k * 3 + 1]]
                aug_image[:, :, 2] = hsi_cube[:, :, band_order[k * 3 + 2]]
                aug_image = aug_image / aug_image.max() * 255
                aug_image = np.uint8(aug_image)
                validation_image_storage(aug_image, np.copy(hsi_gt), training_image_count, training_path)
                training_image_count += 1

        for j in range(500):
            hsi_image = np.array(Image.open(hsi_img_list[0]))
            hsi_cube = X2Cube2(hsi_image)
            hsi_gt = np.array(hsi_gt_list[0])
            band_order = band_selection(hsi_image, hsi_gt)
            h, w, _ = hsi_cube.shape
            for k in range(int(len(band_order) / 3)):
                aug_image = np.zeros((h, w, 3))
                aug_image[:, :, 0] = hsi_cube[:, :, band_order[k * 3 + 0]]
                aug_image[:, :, 1] = hsi_cube[:, :, band_order[k * 3 + 1]]
                aug_image[:, :, 2] = hsi_cube[:, :, band_order[k * 3 + 2]]
                aug_image = aug_image / aug_image.max() * 255
                aug_image = np.uint8(aug_image)
                validation_image_storage(aug_image, np.copy(hsi_gt), validation_image_count, validation_path)
                validation_image_count += 1




