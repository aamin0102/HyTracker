# HyTracker
# Abstract
Hyperspectral images, with their many spectral bands, provide a rich source of material information about an object that can be effectively used for object tracking. However, many trackers in this domain rely on detection-based techniques, which often perform suboptimally in challenging scenarios such as managing occlusions and distinguishing objects in cluttered backgrounds. This underperformance is primarily due to the presence of multiple spectral bands and the inability to leverage this abundance of data for effective tracking. Additionally, the scarcity of annotated hyperspectral videos and the absence of comprehensive temporal information exacerbate these difficulties, further limiting the effectiveness of current tracking methods. To address these challenges, this article introduces the novel Hy-Tracker framework, designed to bridge the gap between hyperspectral data and state-of-the-art object detection methods.
Our approach leverages the strengths of YOLOv7 for object tracking in hyperspectral videos, enhancing both accuracy and robustness in complex scenarios. The Hy-Tracker framework comprises two key components. We introduce a hierarchical
attention for band selection (HAS-BS) that selectively processes and groups the most informative spectral bands, thereby significantly improving detection accuracy. Additionally, we have developed a refined tracker that refines the initial detections by incorporating a classifier and a temporal network using gated recurrent units (GRUs). The classifier distinguishes similar objects, while the temporal network models temporal dependencies across frames for robust performance despite occlusions and scale variations (SVs). Experimental results on hyperspectral benchmark datasets demonstrate the effectiveness of Hy-Tracker in accurately tracking objects across frames and overcoming the challenges inherent in detection-based hyperspectral object tracking (HOT).

# Dataset
HOT_2023/
Project-Name/
│
├── datasets/                           # Main dataset folder
│   ├── training/                       # Training data
│   │   ├── false/                      # False color data
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
│   │   │   └── vis/                    # Visual (RGB) videos
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt
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
│   │   │   └── vis/                    # Visual (RGB) videos
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt
│   │
│   ├── validation/                     # Validation data
│   │   ├── false/
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
│   │   │   └── vis/                    # Visual (RGB) videos
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt
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
│   │   │   └── vis/                    # Visual (RGB) videos
│   │   │       ├── video1/
│   │   │   │   │   ├── img1.jpg        # Image files
│   │   │   │   │   ├── img2.jpg
│   │   │   │   │   ├── ......
│   │   │   │   │   ├── imgn.jpg
│   │   │       │   └── groundtruth_rect.txt


