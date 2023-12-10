
import os
import torch
import pickle
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline,VQDiffusionPipeline,AutoPipelineForText2Image
from inference_utils import save_img_tensor
import argparse
from PIL import Image
import numpy as np
import paintmind as pm


@torch.no_grad()
def sd_image2latent(sd_model, image):
    with torch.no_grad():

        image = (image-0.5)*2
        if len(image.shape) == 3:
            image = image.unsqueeze(0).cuda()
        else:
            image = image.cuda()
        print(image.shape)
        latents = sd_model.vae.encode(image)['latent_dist'].mean
        latents = latents * sd_model.vae.config.scaling_factor

        rec = sd_model.vae.decode(latents/sd_model.vae.config.scaling_factor).sample
        rec = rec*0.5 + 0.5
        save_img_tensor(rec,"rec.png")
    return latents

@torch.no_grad()
def sdxl_image2latent(sd_model, image):
    with torch.no_grad():

        image = (image-0.5)*2
        if len(image.shape) == 3:
            image = image.unsqueeze(0).cuda()
        else:
            image = image.cuda()
        print(image.shape)
        latents = sd_model.vae.encode(image)['latent_dist'].mean
        latents = latents * sd_model.vae.config.scaling_factor

        rec = sd_model.vae.decode(latents/sd_model.vae.config.scaling_factor).sample
        rec = rec*0.5 + 0.5
        save_img_tensor(rec,"rec.png")
    return latents

@torch.no_grad()
def vq_image2latent(sd_model, image):
    with torch.no_grad():
        image = (image-0.5)*2
        if len(image.shape) == 3:
            image = image.unsqueeze(0).cuda()
        else:
            image = image.cuda()
        print(image.shape)
        latents = sd_model.vqvae.encode(image)['latents']
        latents, _, _ = sd_model.vqvae.quantize(latents)
        print("75 models latents",latents)
    return latents

@torch.no_grad()
def vitvqgan_image2latent(sd_model, image):
    with torch.no_grad():

        image = (image-0.5)*2
        if len(image.shape) == 3:
            image = image.unsqueeze(0).cuda()
        else:
            image = image.cuda()
        print(image.shape)
        latents, _, _ = sd_model.encode(image)
        print("latents.shape:",latents.shape)
        rec = sd_model.decode(latents)
        rec = rec*0.5 + 0.5
        save_img_tensor(rec,"rec_encoderdecoder_vitvqgan.png")

    return latents

@torch.no_grad()
def kandinsky_image2latent(sd_model, image):
    with torch.no_grad():
        image = (image-0.5)*2
        if len(image.shape) == 3:
            image = image.unsqueeze(0).cuda()
        else:
            image = image.cuda()
        print(image.shape)
        latents = sd_model.movq.encode(image)['latents']
        latents, _, _ = sd_model.movq.quantize(latents)
        rec = sd_model.movq.decode(latents).sample
        rec = rec*0.5 + 0.5
        save_img_tensor(rec,"rec.png")
    return latents

def get_init_noise(args,model_type,bs=1):
    if model_type in ["sd","sdv21","sdv2base"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        init_noise = torch.randn([bs, args.cur_model.unet.in_channels, height // args.cur_model.vae_scale_factor, width // args.cur_model.vae_scale_factor]).cuda()
        init_noise = torch.cat([sd_image2latent(args.cur_model,args.image0)] * bs).cuda()
        print("init_noise:",init_noise)
    elif model_type in ["sdxl"]:
        height = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        width = args.cur_model.unet.config.sample_size * args.cur_model.vae_scale_factor
        shape = (bs, args.cur_model.unet.config.in_channels, height // args.cur_model.vae_scale_factor, width // args.cur_model.vae_scale_factor)
        init_noise = torch.randn(shape).cuda()
        print("init_noise.shape:",init_noise.shape)
        init_noise = torch.cat([sdxl_image2latent(args.cur_model,args.image0)] * bs).cuda()
        print("init_noise.shape:",init_noise.shape)
    elif model_type in ["vqdiffusion"]:
        embedding_channels = args.cur_model.vqvae.config.vq_embed_dim
        embeddings_shape = (bs, args.cur_model.transformer.height, args.cur_model.transformer.width, embedding_channels)
        init_noise = torch.randn(embeddings_shape).cuda()
        init_noise = torch.cat([vq_image2latent(args.cur_model,args.image0)] * bs).cuda()
    elif model_type in ["vitvqgan"]:
        init_noise = torch.cat([vitvqgan_image2latent(args.cur_model,args.image0)] * bs).cuda()
        #elif model_type in ["kandinsky"]:
    elif model_type in ["kandinsky"]:
        init_noise = torch.cat([kandinsky_image2latent(args.cur_model,args.image0)] * bs).cuda()
    return init_noise

def from_noise_to_image(args,model,noise,model_type):

    if model_type in ["sd","sdv21","sdv2base"]:
        latents = 1 / model.vae.config.scaling_factor * noise
        images = model.vae.decode(latents).sample
        image = (images / 2 + 0.5).clamp(0, 1)
    elif model_type in ["sdxl"]:
        needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast
        latents = noise
        if needs_upcasting:
            model.upcast_vae()
            latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
        image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
        # cast back to fp16 if needed
        if needs_upcasting:
            model.vae.to(dtype=torch.float16)
        image = (image / 2 + 0.5).clamp(0, 1)
    elif model_type in ["vqdiffusion"]:
        latents = noise
        image = model.vqvae.decode(latents, force_not_quantize=True).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    elif model_type in ["vitvqgan"]:
        latents = noise
        image = model.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
    elif model_type in ["kandinsky"]:
        latents = noise
        image = model.movq.decode(latents, force_not_quantize=True)["sample"]
        image = image * 0.5 + 0.5
    return image


def get_model(model_type,model_path,args):
    if model_type in ["sd","sd_unet"]:
        model_id = "runwayml/stable-diffusion-v1-5"
        cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["sdv2base"]:
        model_id = "stabilityai/stable-diffusion-2-base"
        cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["sdv21"]:
        model_id = "stabilityai/stable-diffusion-2-1"
        cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["sdxl"]:
        cur_model = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32
        )
        cur_model = cur_model.to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["vqdiffusion"]:
        cur_model = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float32)
        cur_model = cur_model.to("cuda")
        cur_model.vqvae.eval()

    elif model_type in ["kandinsky"]:
        cur_model = AutoPipelineForText2Image.from_pretrained(
                        "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float32
                    )
        cur_model = cur_model.to("cuda")
        cur_model.unet.eval()
        cur_model.movq.eval()

    elif model_type in ["vitvqgan"]:
        cur_model = pm.create_model(arch='vqgan', version='vit-s-vqgan', pretrained=True).cuda()

    return cur_model
