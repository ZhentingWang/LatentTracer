# LatentTracer

This repository is the source code for [How to Trace Latent Generative Model Generated Images without Artificial Watermark?](https://arxiv.org/pdf/2405.13360) (ICML 2024).

## 🔬Environment
See requirements.txt

The version of Python is 3.10

## 🧰Generating images

For example, generating images using Stable Diffusion v1-5:
```bash
python generate_samples_all.py --arch sd --save_dir ./sdv15_generated_imgs/
```
Generating images using Stable Diffusion v2-base:
```bash
python generate_samples_all.py --arch sdv2base --save_dir ./sdv2base_generated_imgs/
```

## ⚙Reverse-engineering
For example, to conduct reverse-engineering on the belonging images of Stable Diffusion Stable Diffusion v2-base:
```bash
python run_write_re_loss_to_txt.py --gpu 0 --filePath ./sdv2base_generated_imgs/ --model_type sdv2base --lr 0.05 \
--num_iter 100 --write_txt_path ./ReconstructionLosses_Model_sdv2base_Images_sdv2baseGenerated.txt
```

To conduct reverse-engineering on the non-belonging images (images generated by Stable Diffusion v1-5 here):
```bash
python run_write_re_loss_to_txt.py --gpu 0 --filePath ./sdv15_generated_imgs/ --model_type sdv2base --lr 0.05 \
--num_iter 100 --write_txt_path ./ReconstructionLosses_Model_sdv2base_Images_sdv15Generated.txt
```

## 🕵️Belonging detection and visualization the distributions of the reconstruction losses

```bash
python detection.py \
--txt_1 ./ReconstructionLosses_Model_sdv2base_Images_sdv2baseGenerated.txt \
--txt_2 ./ReconstructionLosses_Model_sdv2base_Images_sdv15Generated.txt \
--label_1 'Belongings' \
--label_2 'Non-belongings' \
--threshold 1e-4 \
--save_name ./distribution.pdf
```

Note that the threshold needs to be adjusted for different inspected model based on the distribution of reconstruction losses on its belonging images.

## 🤝Cite this work
You are encouraged to cite the following papers if you use the repo for academic research.

```
@inproceedings{wang2024trace,
  title={How to Trace Latent Generative Model Generated Images without Artificial Watermark?},
  author={Wang, Zhenting and Sehwag Vikash, Chen, Chen and Lyu, Lingjuan and Metaxas, Dimitris N and Ma, Shiqing},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```


```
@inproceedings{wang2023did,
  title={Where Did I Come From? Origin Attribution of AI-Generated Images},
  author={Wang, Zhenting and Chen, Chen and Zeng, Yi and Lyu, Lingjuan and Ma, Shiqing},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


