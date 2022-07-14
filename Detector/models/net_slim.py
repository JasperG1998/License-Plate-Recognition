import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride = 1 ,pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

# class Slim(nn.Module):
#     def __init__(self, cfg = None, phase = 'train'):
#         """
#         :param cfg:  Network related settings.
#         :param phase: train or test.
#         """
#         super(Slim, self).__init__()
#         self.phase = phase
#         self.num_classes = 2

#         self.conv1 = conv_bn(3, 16, 2)
#         self.conv2 = conv_bn(16, 32, 1)
#         self.conv3 = conv_bn(32, 32, 2)
#         self.conv4 = conv_bn(32, 32, 1)
#         self.conv5 = conv_bn(32, 64, 2)
#         self.conv6 = conv_bn(64, 64, 1)
#         self.conv7 = conv_bn(64, 64, 1)
#         self.conv8 = conv_bn(64, 64, 1)

#         self.conv9 = conv_bn(64, 128, 2)
#         self.conv10 = conv_bn(128, 128, 1)
#         self.conv11 = conv_bn(128, 128, 1)

#         self.conv12 = conv_bn(128, 256, 2)
#         self.conv13 = conv_bn(256, 256, 1)

#         self.conv14 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
#             nn.ReLU(inplace=True),
#             conv_bn(64, 256, stride=2, pad=1),
#             nn.ReLU(inplace=True)
#         )
#         self.loc, self.conf, self.landm = self.multibox(self.num_classes)

#     def multibox(self, num_classes):
#         loc_layers = []
#         conf_layers = []
#         landm_layers = []
#         loc_layers += [ conv_bn(64, 3 * 4,   pad=1)]
#         conf_layers += [ conv_bn(64, 3 * num_classes,   pad=1)]
#         landm_layers += [ conv_bn(64, 3 * 8,   pad=1)]

#         loc_layers += [ conv_bn(128, 2 * 4,   pad=1)]
#         conf_layers += [ conv_bn(128, 2 * num_classes,   pad=1)]
#         landm_layers += [ conv_bn(128, 2 * 8,   pad=1)]

#         loc_layers += [ conv_bn(256, 2 * 4,   pad=1)]
#         conf_layers += [ conv_bn(256, 2 * num_classes,   pad=1)]
#         landm_layers += [ conv_bn(256, 2 * 8,   pad=1)]

#         loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
#         landm_layers += [nn.Conv2d(256, 3 * 8, kernel_size=3, padding=1)]
#         return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)


#     def forward(self,inputs):
#         detections = list()
#         loc = list()
#         conf = list()
#         landm = list()

#         x1 = self.conv1(inputs)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x7 = self.conv7(x6)
#         x8 = self.conv8(x7)

#         detections.append(x8)

#         x9 = self.conv9(x8)
#         x10 = self.conv10(x9)
#         x11 = self.conv11(x10)

#         detections.append(x11)

#         x12 = self.conv12(x11)
#         x13 = self.conv13(x12)

#         detections.append(x13)

#         x14= self.conv14(x13)

#         detections.append(x14)

#         for (x, l, c, lam) in zip(detections, self.loc, self.conf, self.landm):
#             loc.append(l(x).permute(0, 2, 3, 1).contiguous())

#             conf.append(c(x).permute(0, 2, 3, 1).contiguous())

#             landm.append(lam(x).permute(0, 2, 3, 1).contiguous())



#         bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
#         classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)
#         ldm_regressions = torch.cat([o.view(o.size(0), -1, 8) for o in landm], 1)


#         print("bbox_regressions :", bbox_regressions.shape)
#         print("classifications :", classifications.shape)
#         print("ldm_regressions :",ldm_regressions.shape)

#         if self.phase == 'train':
#             output = (bbox_regressions, classifications, ldm_regressions)
#         else:
#             output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
#         return output

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0],-1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()


        return out.view(out.shape[0],-1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*8 ,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view( out.shape[0],-1, 8)


class Slim(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Slim, self).__init__()
        self.phase = phase
        self.num_classes = 2

        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_bn(16, 32, 1)
        self.conv3 = conv_bn(32, 32, 2)
        self.conv4 = conv_bn(32, 32, 1)
        self.conv5 = conv_bn(32, 64, 2)
        self.conv6 = conv_bn(64, 64, 1)
        self.conv7 = conv_bn(64, 64, 1)
        self.conv8 = conv_bn(64, 64, 1)

        self.conv9 = conv_bn(64, 64, 2)
        self.conv10 = conv_bn(64, 64, 1)
        self.conv11 = conv_bn(64, 64, 1)

        self.conv12 = conv_bn(64, 64, 2)
        self.conv13 = conv_bn(64, 64, 1)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=64)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=64)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=64)
    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead




    def forward(self,inputs):
        detections = list()
        loc = list()
        conf = list()
        landm = list()

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        detections.append(x8)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)

        detections.append(x11)


        x12 = self.conv12(x11)
        x13 = self.conv13(x12)

        detections.append(x13)
        # print(self.BboxHead[0](detections[0]).shape)
        bbox_A = self.BboxHead[0](detections[0]).permute(0,2,1).contiguous()
        bbox_B = self.BboxHead[1](detections[1]).permute(0,2,1).contiguous()
        bbox_C = self.BboxHead[2](detections[2]).permute(0,2,1).contiguous()
        
        classifications_A  = self.ClassHead[0](detections[0]).permute(0,2,1).contiguous()
        classifications_B  = self.ClassHead[1](detections[1]).permute(0,2,1).contiguous()
        classifications_C  = self.ClassHead[2](detections[2]).permute(0,2,1).contiguous()

        ldm_regressions_A = self.LandmarkHead[0](detections[0]).permute(0,2,1).contiguous()
        ldm_regressions_B = self.LandmarkHead[1](detections[1]).permute(0,2,1).contiguous()
        ldm_regressions_C = self.LandmarkHead[2](detections[2]).permute(0,2,1).contiguous()

        bbox_tmp  = torch.cat((bbox_A , bbox_B) , dim = 2)
        bbox_regressions = torch.cat((bbox_tmp  , bbox_C) ,dim =2)
        
        classifications_tmp = torch.cat((classifications_A , classifications_B) , dim = 2)
        classifications = torch.cat((classifications_tmp , classifications_C),dim =2)
        
        ldm_regressions_tmp = torch.cat((ldm_regressions_A , ldm_regressions_B),dim = 2)
        ldm_regressions = torch.cat((ldm_regressions_tmp , ldm_regressions_C),dim = 2)

        # print("bbox_regressions :", bbox_regressions.shape)
        # print("classifications :", classifications.shape)
        # print("ldm_regressions :",ldm_regressions.shape)

        if self.phase == 'train':
            output = (bbox_regressions.permute(0,2,1).contiguous(), classifications.permute(0,2,1).contiguous(),ldm_regressions.permute(0,2,1).contiguous())
        elif self.phase == 'test':
            print("return test")
            output = (bbox_regressions.permute(0,2,1).contiguous(), F.softmax(classifications.permute(0,2,1).contiguous(), dim=-1), ldm_regressions.permute(0,2,1).contiguous())
        elif self.phase == 'export':
            output = (bbox_regressions.permute(2,0,1).contiguous(), classifications.permute(2,0,1).contiguous() ,ldm_regressions.permute(2,0,1).contiguous())
        return output

if __name__ == "__main__":
    net = Slim(phase='train')
    dump = torch.ones((1,3,224,224))
    out = net(dump)
    # from torchsummary import summary
    # summary(net, (3,640,640) , batch_size=1 ,device="cpu")