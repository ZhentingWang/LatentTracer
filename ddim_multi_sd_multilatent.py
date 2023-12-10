import torch
#from torch.cuda.amp import GradScaler
from inference_utils import SSIMLoss,psnr,lpips_fn,save_img_tensor
from inference_models_multilatent import get_init_noise, get_model,from_noise_to_image
from inference_image0_multilatent import get_image0
#import ot
#from torch_two_sample import SmoothKNNStatistic, SmoothFRStatistic
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
import argparse
import numpy as np
from numba import jit
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument("--input_selection", default="", type=str, help="The path of dev set.")
parser.add_argument("--distance_metric", default="l2", type=str, help="The path of dev set.")
parser.add_argument("--model_type", default="ddpm_cifar10", type=str, help="The path of dev set.")
parser.add_argument("--model_path_", default=None, type=str, help="The path of dev set.")

parser.add_argument("--lr", default=1e-2, type=float, help="")
parser.add_argument("--dataset_index", default=None, type=int, help="")
parser.add_argument("--bs", default=8, type=int, help="")
parser.add_argument("--write_txt_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--num_iter", default=2000, type=int, help="The path of dev set.")
parser.add_argument("--strategy", default="mean", type=str, help="The path of dev set.")
parser.add_argument("--sd_prompt", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_url", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_name", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_type", default=None, type=str, help="The path of dev set.")
parser.add_argument("--input_selection_model_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--re_loss_list_savename", default=None, type=str, help="The path of dev set.")
parser.add_argument("--init_gt_dist", default=None, type=float, help="The path of dev set.")
parser.add_argument("--use_random_init", action="store_true", help="The path of dev set.")
parser.add_argument("--generation_size", default=None, type=int, help="The path of dev set.")

args = parser.parse_args()


@jit(nopython=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass

def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    IJ = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)

    Fij = Counter(IJ).items()
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H

print("point 0-1 memory:",torch.cuda.memory_allocated())

args.cur_model = get_model(args.model_type,args.model_path_,args)
image0, gt_noise = get_image0(args)
image0 = image0.detach()
args.image0 = image0
init_noise = get_init_noise(args,args.model_type,bs=args.bs)
print("main106 init_noise.shape:",init_noise.shape)
print("point 0-2 memory:",torch.cuda.memory_allocated())

if args.generation_size:
    gt_noise = None

print("point 0-3 memory:",torch.cuda.memory_allocated())

print(gt_noise)
init_noise = init_noise[0].unsqueeze(0)
if args.init_gt_dist:
    init_noise = gt_noise + args.init_gt_dist*torch.randn(gt_noise.shape).cuda()
    init_noise_distance2gt = torch.nn.MSELoss(reduction='none')(init_noise,gt_noise).mean()
    print("init_noise_distance2gt:",init_noise_distance2gt)

if args.use_random_init:
    init_noise = torch.randn(init_noise.shape).cuda()

print("init_noise.shape:",init_noise.shape)

if args.model_type in ["sd"]:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)
else:
    cur_noise = torch.nn.Parameter(torch.tensor(init_noise)).cuda()
    optimizer = torch.optim.Adam([cur_noise], lr=args.lr)

if args.distance_metric == "l1":
    criterion = torch.nn.L1Loss(reduction='none')
elif args.distance_metric == "l2":
    criterion = torch.nn.MSELoss(reduction='none')
elif args.distance_metric == "ssim":
    criterion = SSIMLoss().cuda()
elif args.distance_metric == "psnr":
    criterion = psnr
elif args.distance_metric == "lpips":
    criterion = lpips_fn

import time

args.measure = 9999

print("point 1 memory:",torch.cuda.memory_allocated())

re_loss_list = []

for i in range(args.num_iter):
    start_time = time.time()
    print("step:",i)
    print("cur_noise.shape:",cur_noise.shape)
    image = from_noise_to_image(args,args.cur_model,cur_noise,args.model_type)
    loss = criterion(image0.detach(),image).mean()

    epoch_num_str=""
    if i%100==0:
        epoch_num_str=str(i)

    with torch.no_grad():
        save_img_tensor(image,"./imgs/image_cur_"+args.input_selection+"_"+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")

    print(criterion(image0,image).mean(-1).mean(-1).mean(-1))
    min_value = criterion(image0,image).mean(-1).mean(-1).mean(-1).min()
    mean_value = criterion(image0,image).mean()
    print("min: ",min_value)
    print("mean: ",mean_value)

    if (args.strategy == "min") and (min_value < args.measure):
        args.measure = min_value
    if (args.strategy == "mean") and (mean_value < args.measure):
        args.measure = mean_value
    print("measure now:",args.measure)

    if args.distance_metric == "lpips":
        loss = loss.mean()
    print("loss "+args.input_selection+" "+args.distance_metric+":",loss)

    re_loss_list.append(loss.detach().item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if gt_noise is not None:
        noise_distance = torch.nn.MSELoss(reduction='none')(cur_noise,gt_noise)
        print("gt_noise.norm():",gt_noise[0].norm())
        print("noise_distance L2:",noise_distance.mean(-1).mean(-1).mean(-1))
        print("cur_noise.norm():",cur_noise[0].norm())

    if i%50==0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*0.5

    end_time = time.time()
    print("time for one iter: ",end_time-start_time)
    torch.cuda.empty_cache()

print("point 2 memory:",torch.cuda.memory_allocated())

if args.write_txt_path:
    with open(args.write_txt_path,"a") as f:
        f.write(str(args.measure.item())+"\n")

if args.sd_prompt:
    save_img_tensor(image0,"./imgs/ORI_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./imgs/last_"+args.sd_prompt+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
if args.input_selection_url:
    save_img_tensor(image0,"./imgs/ORI_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")
    save_img_tensor(image,"./imgs/last_"+args.input_selection_url.split("/")[-1]+args.distance_metric+"_"+str(args.lr)+"_bs"+str(args.bs)+epoch_num_str+"_"+".png")

if args.input_selection_name:
    img1 = cv2.imread(args.input_selection_name, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (512,512), interpolation=cv2.INTER_AREA)
    t1 = time.time()
    H1 = calcEntropy2dSpeedUp(img1, 3, 3)
    t2 = time.time()
    print("H1:",H1)
    print(t2 - t1, 's')

if args.re_loss_list_savename:
    with open(args.re_loss_list_savename, 'wb') as f:
        pickle.dump(re_loss_list, f)
