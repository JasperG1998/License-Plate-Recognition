import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

class LPDetection(data.Dataset):
    def __init__(self , dataset_root ,  preproc = None , device = 'cpu') -> None:
        super(LPDetection , self).__init__()
        self.dataset_root = dataset_root
        self.device = device
        self.imgs_paths = []
        self.preproc = preproc

        # for dirname in os.listdir(dataset_root):
        #     dir_path  = os.path.join(dataset_root ,dirname)
        #     if not os.path.isdir(dir_path):
        #         continue
        #     for img_name in os.listdir(dir_path):
        #         img_path = os.path.join(dir_path , img_name)
        #         self.imgs_paths.append(img_path)


        for images in os.listdir(dataset_root):
            if images.endswith(".jpg"):
                filepath = os.path.join(dataset_root , images)
                self.imgs_paths.append(filepath)
        random.shuffle(self.imgs_paths)

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, index: int):
        filepath = self.imgs_paths[index]
        img = cv2.imread(filepath)
        basename = os.path.basename(filepath)
        imgname , ext = os.path.splitext(basename)
        bbox_points , landmark_points , LP_number =  self.decode_name(imgname)

        labels =  [bbox_points + landmark_points]
        annotations = np.zeros((0, 13))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 13))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]
            annotation[0, 3] = label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[6]    # l1_x
            annotation[0, 7] = label[7]    # l1_y
            annotation[0, 8] = label[8]   # l2_x
            annotation[0, 9] = label[9]   # l2_y
            annotation[0, 10] = label[10]  # l3_x
            annotation[0, 11] = label[11]  # l3_y
            if (annotation[0, 4]<0):
                annotation[0, 12] = -1
            else:
                annotation[0, 12] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        
        return torch.from_numpy(img), target


    def decode_point_seg(self, bounding_seg):
            points_seg = bounding_seg.split('_')
            points =  []
            for seg in points_seg:
                for value in seg.split('&'):
                    points.append(int(value))
            return points  # [lef_up_x ,left_uy_y, right_down_x , right_down_y]

    def decode_name(self , img_name:str) -> list:
        segments = img_name.split('-')
        # print("segment :",segments)
        bounding =segments[2]
        landmark = segments[3]  #右下角开始顺时针开始

        bbox_points = self.decode_point_seg(bounding)
        landmark_points = self.decode_point_seg(landmark)
        LP_number = [int(value) for value in segments[4].split('_')]
        return bbox_points , landmark_points , LP_number

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
