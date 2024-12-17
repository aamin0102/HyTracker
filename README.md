#  Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos
# Abstract
Hyperspectral images, with their many spectral bands, provide a rich source of material information about an object that can be effectively used for object tracking. However, many trackers in this domain rely on detection-based techniques, which often perform suboptimally in challenging scenarios such as managing occlusions and distinguishing objects in cluttered backgrounds. This underperformance is primarily due to the presence of multiple spectral bands and the inability to leverage this abundance of data for effective tracking. Additionally, the scarcity of annotated hyperspectral videos and the absence of comprehensive temporal information exacerbate these difficulties, further limiting the effectiveness of current tracking methods. To address these challenges, this article introduces the novel Hy-Tracker framework, designed to bridge the gap between hyperspectral data and state-of-the-art object detection methods.
Our approach leverages the strengths of YOLOv7 for object tracking in hyperspectral videos, enhancing both accuracy and robustness in complex scenarios. The Hy-Tracker framework comprises two key components. We introduce a hierarchical
attention for band selection (HAS-BS) that selectively processes and groups the most informative spectral bands, thereby significantly improving detection accuracy. Additionally, we have developed a refined tracker that refines the initial detections by incorporating a classifier and a temporal network using gated recurrent units (GRUs). The classifier distinguishes similar objects, while the temporal network models temporal dependencies across frames for robust performance despite occlusions and scale variations (SVs). Experimental results on hyperspectral benchmark datasets demonstrate the effectiveness of Hy-Tracker in accurately tracking objects across frames and overcoming the challenges inherent in detection-based hyperspectral object tracking (HOT).

Paper link: [Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos](https://ieeexplore.ieee.org/abstract/document/10569013))
`
# Dataset
```
HOT_2023/
Project-Name/
│
├── datasets/                           # Main dataset folder
│   ├── training/                       # Training data
│   │   ├── hsi/                        # Hyperspectral data
│   │   │   ├── nir/                    # NIR videos
│   │   │   │   ├── video1/             # First video in NIR
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │   │   │   └── groundtruth_rect.txt  # Ground truth for bounding boxes
│   │   │   │   ├── video2/             # Second video
│   │   │   ├── rednir/                 # RED-NIR videos
│   │   │   │   ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │   │   │   └── groundtruth_rect.txt
│   │   │   └── vis/                   
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt
│   │
│   ├── validation/                     # Validation data
│   │   ├── hsi/                        # HSI validation
│   │   │   ├── nir/                    # NIR videos
│   │   │   │   ├── video1/             # First video in NIR
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │   │   │   └── groundtruth_rect.txt  # Ground truth for bounding boxes
│   │   │   │   ├── video2/             # Second video
│   │   │   ├── rednir/                 # RED-NIR videos
│   │   │   │   ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │   │   │   └── groundtruth_rect.txt
│   │   │   └── vis/                    
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt
```
# Training
The training of Hy-Tracker consists of three parts:
1. Band Selection using Hierarchical Attention for Band Selection (HASBS):
   
   i. The HOT2023 dataset consists of three different types of data: vis, nir and rednir. Therefore, we develop three HASBS, one for each type of data.
   
   ii. Run the main file of the band15, band16, and band25 folders under the HASBS folder using the appropriate link to the datasets.
3. Sequence model using GRU:
   
   i. Prepare the dataset by running the create_dataset.py under GRU_Network using the appropriate link to the datasets.
   
   ii. Run the training.py under the GRU_Network for the sequence model
5. YOLO training:

   i. Create the dataset for YOLO training by running nir_data_processing, rednir_data_processing and vis_data_processing files under the data_processing folder. We used all the training sets plus the first frame of the validation datasets.

   ii. Download the yolo.pt file from this link: [yolo.pt](https://drive.google.com/file/d/1GfZpbcW_5GQP2WVtt2vFIx-SW8pckVJK/view) and put it in pretrained folder

   ii. Run the training.py file to train the Yolo model.
# Results
|                | AUC    | DP    | link           |
|----------------|------- |-------|----------------|
| HOT 2022       | 0.728  | 0.972 | [HOT 2022](https://github.com/aamin0102/HyTracker/tree/main/Results/HOT2022))   |
| HOT 2023       | 0.642  | 0.847 | [HOT 2023](https://github.com/aamin0102/HyTracker/tree/main/Results/HOT2023)      |
| HOT 2024       | 0.531  | 0.692 | [HOT 2024](https://github.com/aamin0102/HyTracker/tree/main/Results/HOT2024)   |

# If this work is helpful to you, please cite it as:
```bibtex
@article{islam2024hy,
  title={Hy-Tracker: A Novel Framework for Enhancing Efficiency and Accuracy of Object Tracking in Hyperspectral Videos},
  author={Islam, Mohammad Aminul and Xing, Wangzhi and Zhou, Jun and Gao, Yongsheng and Paliwal, Kuldip K},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
