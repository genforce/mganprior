import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, vgg19


def get_loss(loss_name, args):
    if loss_name == 'VGG':
        return VGGLoss(args.vgg_layer, args)
    elif loss_name == 'L1':
        return nn.L1Loss(reduction='mean')
    elif loss_name == 'L2':
        return nn.MSELoss(reduction='mean')
    elif loss_name == 'Combine':
        return CombinationLoss(args)


class CombinationLoss(nn.Module):
    def __init__(self, args):
        super(CombinationLoss, self).__init__()
        self.l1_lambda = args.l1_lambda
        self.l2_lambda = args.l2_lambda
        self.vgg_lambda = args.vgg_lambda
        self.vgg = VGGLoss(args.vgg_layer, args)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def cuda(self, device=None):
        self.l1.cuda()
        self.mse.cuda()
        self.vgg.cuda()

    def forward(self, x, gt):
        l1 = self.l1(x, gt)
        l2 = self.mse(x, gt)
        vgg = self.vgg(x, gt)
        # print(l1.item(), l2.item(), vgg.item())
        return self.l1_lambda * l1 + \
               self.l2_lambda * l2 + \
               self.vgg_lambda * vgg


class VGGLoss(nn.Module):
    """
   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (1): ReLU(inplace)
   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (3): ReLU(inplace)
   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (6): ReLU(inplace)
   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (8): ReLU(inplace)
   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (11): ReLU(inplace)
   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (13): ReLU(inplace)
   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (15): ReLU(inplace)
   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (18): ReLU(inplace)
   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (20): ReLU(inplace)
   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (22): ReLU(inplace)
   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (25): ReLU(inplace)
   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (27): ReLU(inplace)
   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
   (29): ReLU(inplace)
   (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    """
    def __init__(self, vgg_layer, args):
        super(VGGLoss, self).__init__()
        self.vgg = nn.Sequential(*list(vgg16(pretrained=True).children())[0][:vgg_layer])
        if args.vgg_loss_type == 'L2':
            self.loss = F.mse_loss
        elif args.vgg_loss_type == 'L1':
            self.loss = F.l1_loss
        self.pre_processing_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.pre_processing_std = torch.Tensor([0.229, 0.224, 0.225])
        self.resize = args.image_size

    def cuda(self, device=None):
        self.vgg.cuda(device=device)
        self.pre_processing_mean = self.pre_processing_mean.cuda(device=device)
        self.pre_processing_std = self.pre_processing_std.cuda(device=device)

    def forward(self, x, gt):
        """
        :param x: [-1.0, 1.0]
        :param gt: [-1.0, 1.0]
        :return:
        """
        x = (x * 0.5 + 0.5).sub_(self.pre_processing_mean[:, None, None]).div_(self.pre_processing_std[:, None, None])
        gt = (gt * 0.5 + 0.5).sub_(self.pre_processing_mean[:, None, None]).div_(self.pre_processing_std[:, None, None])
        x_features = self.vgg(F.interpolate(x, size=self.resize, mode='nearest'))
        gt_features = self.vgg(F.interpolate(gt, size=self.resize, mode='nearest'))
        return self.loss(x_features, gt_features, reduction='mean') * 0.001

