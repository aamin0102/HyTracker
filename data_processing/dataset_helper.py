import os
import random
import shutil
from scipy import spatial
import cv2
import numpy as np
import albumentations as A


def gen_config(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if(len(gt_data_per_image))==3:
            gt_data_per_image = line.split('\t')

        if len(gt_data_per_image)==5:
            gt_data_per_image = gt_data_per_image[0:4]
        gt_data_int = list(map(float, gt_data_per_image))
        gt_data_int = list(map(int, gt_data_int))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def make_dataset_directory(dataset_path):
    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)


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



def random_flip(image, bounding_box):
    image = np.asarray(image)
    flip_code = random.choice([1, 0])  # -1 for horizontal flip, 0 for vertical flip, 1 for both flips disabled
    flipped_image = cv2.flip(image, flip_code)

    width = image.shape[1]
    height = image.shape[0]

    if flip_code == 1:  # Horizontal flip
        # Adjust the bounding box coordinates
        flipped_bounding_box = bounding_box.copy()
        flipped_bounding_box[0] = image.shape[1] - bounding_box[0] - bounding_box[2]

    elif flip_code == 0:  # Vertical flip
        flipped_bounding_box = bounding_box.copy()
        flipped_bounding_box[1] = image.shape[0] - bounding_box[1] - bounding_box[3]

    return flipped_image, flipped_bounding_box


def rotate_image_with_bounding_box(image, bounding_box):
    image = np.asarray(image)
    # Get the image dimensions
    height, width = image.shape[:2]

    # Randomly select the rotation angle between -45 and 45 degrees
    angle = random.uniform(-45, 45)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation on the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Adjust the bounding box coordinates
    x, y, w, h = bounding_box
    box_center_x = x + w / 2
    box_center_y = y + h / 2

    # Rotate the bounding box center point
    rotated_center = np.dot(rotation_matrix, np.array([[box_center_x], [box_center_y], [1]]))
    rotated_center_x = rotated_center[0, 0]
    rotated_center_y = rotated_center[1, 0]

    # Calculate the rotated bounding box coordinates
    rotated_x = rotated_center_x - w / 2
    rotated_y = rotated_center_y - h / 2

    rotated_bounding_box = [int(rotated_x), int(rotated_y), int(w), int(h)]

    return rotated_image, rotated_bounding_box


def augmented_image(img):
    img = np.array(img)
    t_number = np.random.randint(1, 15)
    brightness = np.random.random()
    contrast = np.random.random()
    saturation = np.random.random()
    hue = np.random.random()
    transform = A.Compose([
        A.ChannelShuffle(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                      always_apply=False, p=0.2),
        A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=90, beta_limit=(0.5, 8.0),
                       noise_limit=(0.9, 1.1), always_apply=False, p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.2),
        A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.2),
        A.ZoomBlur (max_factor=1.31, step_factor=(0.01, 0.03), always_apply=False, p=0.2),
        A.UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=0.2),
        A.Superpixels (p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=0.2),
        A.ToGray(p=0.2),
        A.ToSepia(p=0.2),

        ])

    transformed = transform(image=img)
    transformed_image = transformed['image']
    return transformed_image


def convert_to_yolo(bbox, image_width, image_height):
    x, y, w, h = bbox

    center_x = (x + (w / 2)) / image_width
    center_y = (y + (h / 2)) / image_height
    width = w / image_width
    height = h / image_height

    return center_x, center_y, width, height


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

def crop_and_paste(image, bbox):
    # Extract coordinates of the bounding box
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    # x_min, y_min, x_max, y_max = bbox

    # Crop the bounding box from the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Generate random positions for pasting
    max_x = image.shape[1] - (x_max - x_min)
    max_y = image.shape[0] - (y_max - y_min)
    new_x = random.randint(0, max_x)
    new_y = random.randint(0, max_y)

    # Extract the region where the cropped image will be pasted
    paste_region = image[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)]

    # Calculate the average color of the surrounding area
    surrounding_area = np.copy(image)
    surrounding_area[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)] = 0
    avg_color = np.mean(surrounding_area, axis=(0, 1))

    # Paste the cropped image randomly in another position
    image[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)] = cropped_image

    # Fill the previous position with the average color
    image[y_min:y_max, x_min:x_max] = avg_color

    return image, [new_x, new_y, w, h]