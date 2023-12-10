
import torch

import torchvision
from torchvision import transforms

from piqa import SSIM
import lpips

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)
    
def psnr(img1, img2):
    img1 = img1*255
    img2 = img2*255
    #mse = torch.mean((img1 - img2) ** 2)
    mse = ((img1 - img2)**2).mean(-1).mean(-1).mean(-1)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).mean()

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
def lpips_fn(img1, img2):
    img1 = (img1 - 0.5)*2
    img2 = (img2 - 0.5)*2
    return loss_fn_vgg(img1,img2)

def save_img_tensor(img,name):
    #img = (img / 2 + 0.5).clamp(0, 1)
    torchvision.utils.save_image(img, name)
    #img = img.cpu().permute(0, 2, 3, 1).numpy()
    #img = ddim.numpy_to_pil(img)[0]
    #img.save(name)