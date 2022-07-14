import torch
from model import  build_lprnet
import cv2
import os 
import numpy as np

class Tester(object):
    def __init__(self , images_folder, pretrained , chars ,mean = (0.5, 0.5, 0.5) , std =(128/255.0, 128/255.0, 128/255.0)) -> None:
        self.images_folder = images_folder
        self.mean = mean
        self.std = std 
        self.chars = chars
        images = []
        for image in os.listdir(self.images_folder):
            if image.endswith('.jpg'):
                images.append(os.path.join(self.images_folder , image))
        self.images = images
        self.net = build_lprnet(lpr_max_len=8 ,phase='test',class_num=68 , dropout_rate=0)
        self.net.load_state_dict(torch.load(pretrained))
        self.net.eval()

    def transform(self, img):
        img = img.astype(np.float32)
        img /= 255.
        img -= np.array((0.5, 0.5, 0.5) , dtype=np.float32)
        img /= np.array((128/255.0, 128/255.0, 128/255.0) , dtype=np.float32)
        img = img[:, :, (2, 1, 0)]  #BGR 2 RGB
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        return img# HWC 2 CHW

    def LP_number2chars(self , LP_number):
        '''
        the function aim to convert image name to trainning labels
        args:
            LP_number : the PL number of image name 
        '''
        chars  =   ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
                            "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
                            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
                            "新", "警", "学",
                            "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
        LP_string = ''
        for idx in LP_number:
            LP_string += chars[idx]
        return LP_string

    def drop_repeate_and_blank(self, prebs):
        preb = prebs[0, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(self.chars) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(self.chars) - 1):
                if c == len(self.chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        return no_repeat_blank_label

    def run(self):
        for filepath in self.images:
            image = cv2.imread(filepath)
            image = self.transform(image)
            output = self.net(image)
            no_repeat_blank_label = self.drop_repeate_and_blank(output.detach().numpy())
            string_plr = self.LP_number2chars(no_repeat_blank_label)
            print("the {} image is {} ,  {}".format(os.path.basename(filepath) ,  string_plr , ))

            if 


if __name__ == "__main__":
    images_folder = '/home/bits/Trainer/License_Plate_Recognition/Recognizer2/data/test'
    pretrained = "/home/bits/Trainer/License_Plate_Recognition/Recognizer2/weights/LPRNet__iteration_4000.pth"
    chars  =   ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
                            "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
                            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
                            "新", "警", "学",
                            "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
    tester = Tester(images_folder= images_folder , pretrained= pretrained , chars= chars)
    tester.run()