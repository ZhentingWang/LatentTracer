import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=str, default='')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--reference_model', type=str, default='cm_cd_lpips')
parser.add_argument('--reference_bs', type=str, default='2')
parser.add_argument('--filePath', type=str, default='2')
parser.add_argument('--num_iter', type=int, default=80)
parser.add_argument('--write_txt_path', type=str, default='./run_sd_generated_sdv15.txt')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument("--use_random_init", action="store_true", help="The path of dev set.")
parser.add_argument('--model_type', type=str, default='')
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()


name_list = os.listdir(args.filePath)

if os.path.exists(args.write_txt_path):
  os.remove(args.write_txt_path)

counter = 0
for name in name_list:
    print(name)
    if counter ==args.num_samples:
        break
    input_content = args.filePath + name
    print("input_content:",input_content)

    if args.use_random_init:
        cmd = 'CUDA_VISIBLE_DEVICES={} python ddim_multi_sd_multilatent.py --model_type {} --input_selection_name {} --write_txt_path {} \
            --distance_metric l2 --bs 1 --num_iter {} --strategy min --lr {} --use_random_init'.format(args.gpu,args.model_type,input_content,args.write_txt_path,args.num_iter,args.lr)
    else:
        cmd = 'CUDA_VISIBLE_DEVICES={} python ddim_multi_sd_multilatent.py --model_type {} --input_selection_name {} --write_txt_path {} \
            --distance_metric l2 --bs 1 --num_iter {} --strategy min --lr {}'.format(args.gpu,args.model_type,input_content,args.write_txt_path,args.num_iter,args.lr)
    counter = counter + 1

    os.system(cmd)