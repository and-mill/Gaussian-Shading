import argparse
import copy
from tqdm import tqdm
import json
import torch
import os
from statistics import mean, stdev
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from image_utils import *
from pytorch_fid.fid_score import *
from watermark import *

import pandas as pd


def main(args):
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # dataset
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_files = dataset['images']
        annotations = dataset['annotations']
        prompt_key = 'caption'

    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    w_dir = f'./fid_outputs/coco/{args.run_name}/w_gen'
    os.makedirs(w_dir, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    # main loop
    rows = []
    for i in tqdm(range(0, args.num)):
        seed = i + args.gen_seed
        
        # image files
        image_id_in_image = image_files[i]['id']
        image_id_padded = image_files[i]['file_name'].split(".")[0]

        # annotations
        image_id_in_annotation = annotations[i]['image_id']
        current_prompt = annotations[i][prompt_key]
        #id_annotation = annotations[i]['id']  # this one does not matter

        # gt image name and path
        gt_image_file_name = image_files[i]['file_name']
        gt_image_path = f'{args.gt_folder}/{gt_image_file_name}'
        # wm image name and path
        wm_image_file_name = image_files[i]['file_name']
        wm_image_path = f'{w_dir}/{wm_image_file_name}'

        # retrieve args
        message = torch.randint(0, 2, [1, 4 // watermark.ch, 64 // watermark.hw, 64 // watermark.hw]).cuda() if args.fix_message else None
        key = get_random_bytes(32) if args.fix_key else None
        nonce = get_random_bytes(12) if args.fix_nonce else None

        # get latent
        set_random_seed(seed)
        if args.watermark:
            init_latents_w, key, nonce, message = watermark.create_watermark_and_return_all(message=message, key=key, nonce=nonce)
        else:
            message = ""
            key = ""
            nonce = ""
            init_latents_w = None

        #outputs = pipe(
        #    current_prompt,
        #    num_images_per_prompt=args.num_images,
        #    guidance_scale=args.guidance_scale,
        #    num_inference_steps=args.num_inference_steps,
        #    height=args.image_length,
        #    width=args.image_length,
        #    latents=init_latents_w,
        #)
        #image_w= outputs.images[0]
        #image_file_name = image_files[i]['file_name']
        #image_w.save(f'{w_dir}/{image_file_name}')

        # save meta
        row = {"index": i,
               
               "watermark_yes_no": args.watermark,
               
               #"id_annotation": id_annotation,

               "image_id_in_image": image_id_in_image,
               "image_id_in_annotation": image_id_in_annotation,

               "image_id_padded": image_id_padded,

               "prompt": current_prompt,
               "seed": seed,

               "gt_image_file_name": gt_image_file_name,
               "gt_image_path": gt_image_path,
               "wm_image_file_name": wm_image_file_name,
               "wm_image_path": wm_image_path,
               
               "message": message,
               "key": key,
               "nonce": nonce,}
        rows.append(row)

        # save every step
        df = pd.DataFrame(rows)
        df.to_csv(args.output_path + 'index.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--run_name', default='Gaussian_Shading')
    parser.add_argument('--num', default=5000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--prompt_file', default='./fid_outputs/coco/meta_data.json')
    parser.add_argument('--gt_folder', default='./fid_outputs/coco/ground_truth')
    parser.add_argument('--output_path', default='./output/1/')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--no_chacha', dest="chacha", action='store_false', default=True, help='chacha20 for cipher')

    # ADDED
    parser.add_argument('--no_watermark', dest="watermark", action='store_false', default=True, help='chacha20 for cipher')
    parser.add_argument('--fix_message', action="store_true", default=False)
    parser.add_argument('--fix_key', action="store_true", default=False)
    parser.add_argument('--fix_nonce', action="store_true", default=False)
    # case 1: no watermark
    # case 2: watermark, fix message, fix key, fix nonce
    # case 3: watermark, fix message, fix key
    # case 4: watermark,              fix key, fix nonce

    
    # FIXED (was missing)
    parser.add_argument('--fpr', default=1e-6, type=float)
    parser.add_argument('--user_number', default=100 * 1000, type=float)

    args = parser.parse_args()

    main(args)
