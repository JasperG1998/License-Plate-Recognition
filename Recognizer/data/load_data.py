from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
                    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
                    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
                    "新", "警", "学",
                    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for image in os.listdir(img_dir):
            if not image.endswith('.jpg'):
                continue
            filepaht = f"{img_dir}/{image}"
            self.img_paths.append(filepaht)
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname , _ = os.path.splitext(basename)
        label =  self.decode_name(imgname)  # number is the idx of chars


        # if len(label) == 8:
        #     if self.check(label) == False:
        #         print(imgname)
        #         assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype(np.float32)
        img /= 255.
        img -= np.array((0.5, 0.5, 0.5) , dtype=np.float32)
        img /= np.array((128/255.0, 128/255.0, 128/255.0) , dtype=np.float32)
        img = img[:, :, (2, 1, 0)]  #BGR 2 RGB
        return img.transpose(2, 0, 1)

    # def check(self, label):
    #     if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
    #             and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
    #         print("Error label, Please check!")
    #         return False
    #     else:
    #         return True

    def decode_point_seg(self, bounding_seg):
        points_seg = bounding_seg.split('_')
        points =  []
        for seg in points_seg:
            for value in seg.split('&'):
                points.append(int(value))
        return points  # [lef_up_x ,left_uy_y, right_down_x , right_down_y]

    def decode_name(self , img_name:str) -> list:
        segments = img_name.split('-')
        LP_number = [int(value) for value in segments[4].split('_')]
        lp = []
        for i , v in enumerate(LP_number):
            if i == 0:
                lp.append(v)
            else:
                lp.append(v+33)
        return lp