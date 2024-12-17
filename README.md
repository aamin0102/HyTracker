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
