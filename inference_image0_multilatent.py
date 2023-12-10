
import torch
import numpy as np
from PIL import Image,ImageFilter
import cv2
from torchvision import transforms
import random
import os
from inference_utils import save_img_tensor
from inference_models_multilatent import get_init_noise,from_noise_to_image
from piqa import SSIM
import io

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

def text2img_get_init_image(args):
    
    if args.model_type in ["sd","sdv21","sdv2base","sdxl","vqdiffusion","kandinsky"]:
        if args.sd_prompt:
            prompt = args.sd_prompt
        else:
            prompt = "(Very sharp)) Portrait Half-body photo of a beautiful raw photo, DSLR, Bokeh, A vibrant regal blonde ((Goblin )) queen with pointy elvish ears, unique face, golden makeup, silver and diamond jewellery , in unique haute couture gown by Chanel, with auburn hair, in a lush ((surreal fantasy )) fairytale, fireflies, (((magical flowers))), . hyper detailed , 8 k resolution, mid day , 8k ((textured skin)), ((Vellus hair)), (chapped lips), freckled cheekbones , catch lights in eyes, ((imperfect skin)), rosacea, remarkable detailed pupils, (( dull skin noise)), ((stretch marks)) (((visible skin detail))), (((skin fuzz))), ((cellulite)) (dry skin)"
           
        if args.generation_size:
            latents = args.cur_model(prompt, num_inference_steps=50, guidance_scale=7.5, height=args.generation_size, width=args.generation_size, output_type="latent",return_dict=False)
        else:
            latents = args.cur_model(prompt, num_inference_steps=50, guidance_scale=7.5,output_type="latent",return_dict=False)
        latents = latents[0]
        print("image0 latents.shape:",latents.shape)
        image = from_noise_to_image(args,args.cur_model,latents,args.model_type).cuda()
        print("sd init image max:",image.max())
    else:
        print("model_type error")
    return image,latents

def get_image0(args):
    gt_noise = None
    
    if args.input_selection == "use_generated_image0":
        with torch.no_grad():
            if args.model_type in ["sd","sdv21","sdv2base","sdxl","vqdiffusion","kandinsky"]:
                image0,gt_noise = text2img_get_init_image(args)
                print("gt_noise:",gt_noise)

            elif args.model_type in ["vitvqgan"]:
                shiba_img = cv2.imread("./0818_sd_generated_imgs/29.png")
                b,g,r = cv2.split(shiba_img)
                shiba_img = cv2.merge([r, g, b])
                shiba_img = cv2.resize(shiba_img, (256,256), interpolation=cv2.INTER_AREA)
                shiba_img = shiba_img/255
                shiba_img = torch.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
                image0 = shiba_img

                image = (image0-0.5)*2
                if len(image.shape) == 3:
                    image = image.unsqueeze(0).cuda()
                else:
                    image = image.cuda()
                print(image.shape)
                latents, _, _ = args.cur_model.encode(image)
                print("latents.shape:",latents.shape)

                rec = args.cur_model.decode(latents)
                rec = rec*0.5 + 0.5
                image0 = rec
            else:
                gt_noise = get_init_noise(args,args.model_type)[0].unsqueeze(0)
                image0 = from_noise_to_image(args,args.cur_model,gt_noise,args.model_type)
            
            save_img_tensor(image0,"image0.png")

    if args.input_selection_name != None:
        shiba_img = cv2.imread(args.input_selection_name)
        b,g,r = cv2.split(shiba_img)
        shiba_img = cv2.merge([r, g, b])
        shiba_img = cv2.resize(shiba_img, (512,512), interpolation=cv2.INTER_AREA)
        print(shiba_img.shape)
        shiba_img_show = Image.fromarray(shiba_img)
        shiba_img_show.save("input_selection_name_img_show3.jpg")
        shiba_img = shiba_img/255
        shiba_img = torch.from_numpy(shiba_img).cuda().clamp(0, 1).permute(2,0,1).unsqueeze(0).float()
        image0 = shiba_img

    if args.input_selection != "use_generated_image0" and args.model_type in ["sd","sdv21","sdv2base","sdxl","vqdiffusion"]:
        if args.model_type in ["sd","sdv21","sdv2base","sdxl"]:
            height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
            width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor

        elif args.model_type in ["vqdiffusion"]:
            height = args.cur_model.transformer.height
            width = args.cur_model.transformer.width
        image0 = transforms.Resize(height)(image0)
        save_img_tensor(image0,"image0_sd_not_generated.png")

    if args.model_type in ["sd","sdv2base","kandinsky"]:
        imsize = 512
    elif args.model_type in ["sdv21"]:
        imsize = 768
    elif args.model_type in ["sdxl"]:
        imsize = 1024
    elif args.model_type in ["vqdiffusion","vitvqgan"]:
        imsize = 256
    
    image0 = transforms.Resize((imsize,imsize))(image0)
    save_img_tensor(image0,"image0_final.png")

    return image0, gt_noise




