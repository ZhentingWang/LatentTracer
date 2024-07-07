
import os
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline,AutoPipelineForText2Image,VQDiffusionPipeline
import torch


import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--arch', type=str, default='')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--opt', type=str, default="prompthero")

args = parser.parse_args()


if args.arch=="sdxl":
    cur_model = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32
    )
    cur_model = cur_model.to("cuda")
    cur_model.unet.eval()
    cur_model.vae.eval()
elif args.arch=="sd":
    model_id = "runwayml/stable-diffusion-v1-5"
    cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
    cur_model.unet.eval()
    cur_model.vae.eval()
elif args.arch=="sdv2base":
    model_id = "stabilityai/stable-diffusion-2-base"
    cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
    cur_model.unet.eval()
    cur_model.vae.eval()
elif args.arch=="sdv21":
    model_id = "stabilityai/stable-diffusion-2-1"
    cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
    cur_model.unet.eval()
    cur_model.vae.eval()
elif args.arch=="kandinsky":
    cur_model = AutoPipelineForText2Image.from_pretrained(
                        "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float32)
    cur_model = cur_model.to("cuda")
elif args.arch=="vqdiffusion":
    cur_model = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float32)
    cur_model = cur_model.to("cuda")

prompt_list = ["cyber punk robot, dark soul blood borne boss, face hidden, RTX technology, high resolution, light scttering",
            "RAW photo of young woman in sun glasses sitting on beach, (closed mouth:1.2), film grain, high quality, Nikon D850, hyperrealism, photography, (realistic, photo realism:1. 37), (highest quality) <lora:polyhedron_skinny_all:0.7> <lora:more_details:0.3>",
            "full color portrait of bosnian (bosniak) woman wearing a keffiyeh, epic character composition, by ilya kuvshinov, terry richardson, annie leibovitz, sharp focus, natural lighting, subsurface scattering, f2, 35mm, film grain, award winning,  8k",
            "Slim sexy and tiny elf girl in a reflective armor in a luminous backlit glowing mushroom wonderland, running, highly detailed face, perfect skin, young looking face, blond hair, elegant, perfectly lit body and face, drops, glowing particles, neon lights around the huge mushroom, faded, hyperdetailed, digital art, complex background, glowing skin",
            "stained glass art of goddess, mosaic-stained glass art, stained-glass illustration, close up, portrait, concept art, (best quality, masterpiece, ultra-detailed, centered, extremely fine and aesthetically beautiful, super fine illustration), centered, epic composition, epic proportions, intricate, fractal art, zentangle, hyper maximalism",
            "An artistic composition featuring a person with orange hair casually squatting in a park, capturing their nonchalant expression, hot pants, The focus is on their unique sense of style, particularly their white panties peeking out from under their attire.",
            "a woman holding a glowing ball in her hands, featured on cgsociety, fantasy art, very long flowing red hair, holding a pentagram shield, looks a bit similar to amy adams, lightning mage spell icon, benevolent android necromancer, high priestess tarot card, anime goddess, portrait of celtic goddess diana, featured on artstattion",
            "masterpiece, girl alone, solo, incredibly absurd, hoodie, headphones, street, outdoor, rain, neon,",
            "(best quality) Detailed sportswear girl, sitting on the gym floor, spread legs, wearing an intricate leggings and sports bra, highlighting the slight abs, while the chest tattoo adds a cinematic touch to the scene amidst the black tattoo, and cinematic lighting perfects the overall ambiance.",
            "(8k, RAW photo, high sensitivity, best quality, masterpiece, ultra high resolution, fidelity: 1.25), upper body, cat ears, (night), rain, walk, city lights, delicate face, wet white shirt",
            "masterpiece, centered, dynamic pose, 1girl, cute, calm, intelligent, red wavy hair, standing, batik swimsuit, beach background,",
            "masterpiece, award winning, best quality, high quality, extremely detailed, cinematic shot, 1girl, adventurer, riding on a dragon, fantasy theme, HD, 64K",
            "((masterpiece:1.4, best quality))+, (ultra detailed)+, blue hair , wolfcut, pink eyes, 1 girl,cyberpunk city,flat chest,wavy hair,mecha clothes,(robot girl),cool movement,silver bodysuit,colorful background,rainy days,(lightning effect),silver dragon armour,(cold face),cowboy shot,",
            "masterpiece, centered, concept art, wide shot, art nouveau, skyscraper, architecture, modern, sleek design, photography, raw photo, sharp focus, vibrant illustrations, award winning, <lora:EMS-6033-EMS:0.8>, <lora:EMS-179-EMS:0.6>",
            "masterpiece, best quality, mid shot, front view, concept art, 1girl, warrior outfit, pretty, medium blue wavy hair, walking, curious, exploring city, london city street background, Fantasy theme, depth of field, global illumination, (epic composition, epic proportion), Award winning, HD, Panoramic,",
            "a couple of women standing next to each other holding candles, inspired by WLOP, cgsociety contest winner, ancient libu young girl, 4 k detail, dressed in roman clothes, lovely detailed faces, loli, high detailed 8 k, twin souls, 🌺 cgsociety, beautiful maiden",
            "Tigrex from monster hunter, detailed scales, detailed eyes, anatomically correct, UHD, highly detailed, raytracing, vibrant, beautiful, expressive, masterpiece, oil painting",
            "Fashion photography of a joker, 1800s renaissance, clown makeup, editorial, insanely detailed and intricate, hyper-maximal, elegant, hyper-realistic, warm lighting, photography, photorealistic, 8k",
            "octane render of cyberpunk batman by Tsutomu nihei, chrome silk with intricate ornate weaved golden filiegree, dark mysterious background --v 4 --q 2",
            "a cat with a (pirate hat:1.2) on a tropical beach, ~*~Enhance~*~ , in the style of Clyde Caldwell, vibrant colors, ",
            "masterpiece, portrait, medium shot, cel shading style, centered image, ultra detailed illustration of Hatsune Miku of cool posing, inkpunk, ink lines, strong outlines, bold traces, unframed, high contrast, cel-shaded, vector, 32k resolution, best quality",
            "((A bright vivid chaotic cyberpunk female, Fantastic and mysterious, full makeup, blue sky hair, (nature and magic), electronic eyes, fantasy world))",
            "broken but unstoppable masked samurai in full battle gear, digital illustration, brutal epic composition, (expressionism style:1. 1), emotional, dramatic, gloomy, 8k, high quality, unforgettable, emotional depth",
            "studio lighting, film, movie scene, extreme detail, 12k, masterpiece, hyperrealistic, realistic, Canon EOS R6 Mark II, a dragon made out of flowers and leaves, beautiful gold flecks, colorful paint, golden eye, detailed body, detailed eye, multiple colored flowers",
            "Photo realistic young Farscape Chiana, kissy face, full Farscape Chiana white face paint, black shadowy eye makeup, white/gray lips, close-up shot, thin, fit, Fashion Pose, DSLR, F/2. 8, Lens Flare, 5D, 16k, Super-Resolution, highly detailed, cinematic lighting",
            "A retro vintage Comic style poster, of a post apocalyptic universe, of a muscle car, extreme color scheme, action themed, driving on a desert road wasteland, fleeting, chased by a giant fire breathing serpent like fantasy creature, in action pose, highly detailed digital art, jim lee",
            "cinematic CG 8k wallpaper, action scene from GTA V game, perfect symmetric cars bodies and elements, wheels rotating, real physics based image, extremely detailed 4k digital painting (design trending on (Agnieszka Doroszewicz), Behance, Andrey Tkachenko, GTA V game, artstation, BMW X6 realistic design",
            "the Hulk in his Worldbreaker form, his power and rage reach astronomical levels, amidst a cityscape in ruins, reflecting the destruction he can unleash",
            "masterpiece, portrait, medium shot, cel shading style, centered image, ultra detailed illustration of Hatsune Miku of cool posing, inkpunk, ink lines, strong outlines, bold traces, unframed, high contrast, cel-shaded, vector, 32k resolution, best quality",
            "Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet.",
            "A photo of a very intimidating orc on a battlefield, cinematic, melancholy, dynamic lighting, dark background, <lora:ClassipeintXL1. 9:1>",
            "A dark fantasy devil predator, photographic, ultra detail, full detail, 8k best quality, realistic, 8k, micro intricate details",
            "Hello darkness, my old friend, I've come to talk to you again, heart-wrenching composition, digital painting, (expressionism:1. 1), (dramatic, gloomy, emotionally profound:1. 1), intense and brooding dark tones, exceptionally high quality, high-resolution, leaving an indelible and haunting impression on psyche, unforgettable, masterpiece",
            "epic, masterpiece, alien friendly standing on moon, intricated organic neural clothes, galactic black hole background, {expansive:2} hyper realistic, octane, ultra detailed, 32k, raytracing",
            "Geometrical art of autumn landscape, warm colors, a work of art, grotesque, Mysterious",
            "a girl with face painting and a golden background is wearing makeup, absurd, creative, glamorous surreal, in the style of zbrush, black and white abstraction, daz3d, porcelain, striking symmetrical patterns, close-up --ar 69:128 --s 750",
            "Forest, large tree, river in the middle, full blue moon, star's, night , haze, ultra-detailed, film photography, light leaks, Larry Bud Melman, trending on artstation, sharp focus, studio photo, intricate details, highly detailed, by greg rutkowski",
            "a futuristic spacecraft winging through the sky, orange and beige color, in the style of realistic lifelike figures, ravencore, hispanicore, liquid metal, greeble, high definition, manticore, photo, digital art, science fiction --v 5. 2",
            "a close up of a person with a sword, a character portrait by Hasegawa Settan, featured on cg society, antipodeans, reimagined by industrial light and magic, sabattier effect, character",
            "Dystopian New York, gritty, octane render, ultra-realistic, cinematic --ar 68:128 --s 750 --v 5. 2",
            "ALIEN SPACECRAFT, WRECKAGE, CRASH, PLANET DESERT, ultra-detailed, film photography, light leaks, trending on artstation, sharp focus, studio photo, intricate details, highly detailed, by greg rutkowski",
            "an expressionist charcoal sketch by Odilon Redon, drawing, face only, a gorgeous Japanese woman, hint of a smile, noticeable charcoal marks, white background, no coloring, no color --ar 69:128 --s 750 --v 5. 2",
            "a man in a futuristic suit with neon lights on his face, cyberpunk art by Liam Wong, cgsociety, computer art, darksynth, synthwave, glowing neon",
            "The image features a bird perched on a branch, dressed in a suit and tie. The bird is holding a cup of hot coffee, in its beak. The coffee cup emits smoke. The scene is quite unusual and whimsical. The bird's attire and the presence of the cup create a sense of humor and playfulness in the image.",
            "carnage, a formidable supervillain, symbiote, bloody, psychopathic, unstoppable, mad, sharp teeth, epic composition, dramatic, gloomy, in the style of mike deodato, realistic detail, realistic hyper-detailed rendering, realistic painted still lifes, insanely intricate",
            "A disoriented astronaut, lost in a galaxy of swirling colors, floating in zero gravity, grasping at memories, poignant loneliness, stunning realism, cosmic chaos, emotional depth, 12K, hyperrealism, unforgettable, mixed media, celestial, dark, introspective",
            "an abstract painting of a beautiful girl, in the style of Pablo Picasso, masterpiece, highly imaginative, dada, salvador dali, i can't believe how beautiful this is, intricate --ar 61:128 --s 750 --v 5. 2",
            "made by Emmanuel Lubezki, Daniel F Gerhartz, character of One Piece movie, Monkey D. Luffy, in straw hat, cinematic lighting, concept photoart, 32k, photoshoot unbelievable half-length portrait, artificial lighting, hyper detailed, realistic, figurative painter with intricate details, divine proportion, sharp focus, Mysterious",
            "a very detailed image of a female cyborg, half human, half machine, very detailed, with cables, wires, mechanical elements in the head and body, dynamic light, glowing electronics, 4 k, inspired by H. r. Giger and Jean ansell and justin Gerard, photorealistic",
            "A beautiful photo of an lion that got lost in the amazon rainforest, rain, mist, 8k, sharp intricate details, masterpiece, imaginative, raytracing, octane render, studio lighting, professionally shot nature photo, godrays, hyperrealistic, ultra high quality, realism, wet, dripping water, wandering through the undergrowth",
            "magic realism, photograph shot with Kodak Portra 800, a Hasselblad 500C, 55mm f/ 1. 8 lens, extreme depth of field, available light, high contrast, Ultra HD, HDR, DTM, 8K style by Eyvind Earle",
            "A silhouette of a woman of dark fantasy standing on the ground, in the style of dark navy and dark emerald, pigeoncore, heavily textured, avian - themed, realistic figures, medieval - inspired, photorealistic painting",
            "first landing on europa moon",
            "Halvard, druid, Spring, green, yellow, red, vibrant, wild, wildflowers masterpiece, shadows, expert, insanely detailed, 4k resolution, intricate detail, art inspired by diego velazquez eugene delacroix",
            ]


save_folder = "./"+args.arch+"_generated_imgs/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

counter = 0
for prompt in prompt_list:
    for i in range(10):
        print("start prompt ", counter)
        print(prompt)
        image = cur_model(prompt, num_inference_steps=50, guidance_scale=7.5,output_type="pil",return_dict=False)
        image = image[0][0]
        path = save_folder+str(counter)+"_"+str(i)+".png"
        image.save(path)

    print("end prompt ", counter)
    counter=counter+1
