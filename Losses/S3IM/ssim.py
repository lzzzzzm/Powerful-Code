import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel, sigma=None):
    if sigma == None:
        sigma = 0.3 * ((window_size - 1)*0.5 -1)+0.8
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True, stride=1):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel, stride=stride)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    # ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = (2*sigma12)/(sigma1_sq+sigma2_sq)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim = _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)
        return 1 - ssim


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1, stride=None):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        if stride == None:
            self.stride= window_size // 2

        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim3d = _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)
        return 1 - ssim3d

if __name__ == '__main__':
    loss_fn = SSIM()

    img_1 = torch.rand(size=(2, 18, 200, 200))
    img_2 = torch.randint(size=(2, 200, 200), low=0, high=18)
    img_2 = F.one_hot(img_2).to(torch.float).permute(0, 3, 1, 2)

    loss = loss_fn(img_1, img_2)
    print(loss)

    loss_fn_3d = SSIM3D(window_size=16)

    img_1 = torch.rand(size=(1, 18, 200, 200, 16))
    bs, num_clas, h, w, z = img_1.shape

    img_1 = F.softmax(img_1, dim=1)
    img_2 = torch.randint(size=(1, 200, 200, 16), low=0, high=18)
    img_2 = F.one_hot(img_2).to(torch.float).permute(0, 4, 1, 2, 3)

    weights = torch.ones(size=(18, 1, 8, 8, 8), device=img_2.device)
    targ_local_prob = F.conv3d(img_2, weights, groups=18, stride=4)
    pred_local_prob = F.conv3d(img_1, weights, groups=18, stride=4)

    targ_local_prob = targ_local_prob.permute(0, 2, 3, 4, 1).reshape(bs, -1, num_clas) / (8*8*8)
    pred_local_prob = pred_local_prob.permute(0, 2, 3, 4, 1).reshape(bs, -1, num_clas) / (8*8*8)

    loss = F.kl_div(torch.log(pred_local_prob), targ_local_prob, reduction='batchmean')
    print(loss)
    # loss_3d = loss_fn_3d(img_1, img_2)
    # print(loss_3d)