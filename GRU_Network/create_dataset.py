import os

directory = "D:\\hsi_2023_dataset\\training\\hsi"
video_types = os.listdir(directory)
k = 6 # length of the sequences
counter = 1
dataset = "dataset_" + str(k)
os.makedirs(dataset)
for video_type in video_types:
    videos = os.listdir(os.path.join(directory, video_type))
    for video in videos:
        file_name = os.path.join(directory,video_type, video, "groundtruth_rect.txt")
        file = open(file_name, "r")
        lines = file.readlines()
        for i in range(len(lines) - k):
            new_file_name = os.path.join(dataset, str(counter).zfill(6) + '.txt')
            new_file = open(new_file_name, "x")
            for j in range(k+1):
                new_file.write(lines[i+j])
            new_file.close()
            counter = counter + 1