# RobustDistiller

**tl;dr**: Robust distillation recipe for self-supervised speech representation learning (S3RL) models that tackle jointly model compression and robustness against environmental artifacts (noise and reverberation).

*In this repository, you will find useful codes to train/run the model and links to pre-trained weights.*

# Overview

This repository contains codes and artifacts from two papers:

1. [ICASSP 2023] RobustDistiller: Compressing Universal Speech Representations for Enhanced Environment Robustness

2. [Journal Extension - Under Review] An Efficient End-to-End Approach to Noise Invariant Speech Features via Multi-Task Learning

**Problem statement**: Self-supervised speech representation learning enables the extraction of meaningful factors from raw waveforms. These features can then be efficiently used across multiple downstream tasks. However, two significant issues arise when considering the deployment of such methods ``in-the-wild": (i) Their large size, which can be prohibitive for edge applications; and (ii) their robustness to detrimental factors, such as noise and/or reverberation, that can heavily degrade the performance of such systems.

**Our proposal**

> We propose a novel knowledge distillation mechanism, namely RobustDistiller, to tackle both problems jointly. 
Two main modifications are proposed to improve the robustness of the student models: (i) a **feature-denoising knowledge distillation** step that induces the student model to learn noise-invariant representations; and (ii) a **multi-task learning approach via a signal enhancement** step where, given the last hidden state from the student model obtained from a noisy signal, we reconstruct the clean waveform or STFT of the clean input.

![alt text](https://github.com/Hguimaraes/robustdistiller/blob/main/assets/model_arch.png)

Initially, in the ICASSP paper, we only evaluated RobustDistiller on 3 downstream tasks. Later, on a journal extension, we evaluated it over 12 downstream tasks, and it has been shown to outperform several benchmarks regardless of noise type, noise level, and reverberation times. 

Lastly, we show that our recipe is general enough to adapt straightforwardly to other distillation methodologies, such as the recent **DPWavLM**.

# Usage

## Training the upstream model

The first step here is to download the data to train the model. Herein, we rely on different datasets:

1. LibriSpeech (960h) [[Link](https://www.openslr.org/12)]
2. Musan [[Link](https://www.openslr.org/17/)]
3. UrbanSound8K [[Link](https://urbansounddataset.weebly.com/urbansound8k.html)]
4. impulse_responses_000 from DNS4 [[Link](https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/datasets_fullband.impulse_responses_000.tar.bz2)]

The rest of the process is quite similar to training the DistilHuBERT model from S3PRL, and most of our code is adapted from there. We first generate the librispeech metadata as:

```bash
python preprocess/generate_len_for_bucket.py -i $SLURM_TMPDIR/LibriSpeech/
```
Edit the file *config_runner.yaml* from the pretrain/robust_distiller folder. There are several variables that you can experiment with, but to change the file paths, these are the most important for you:

- **libri_root**: /path/to/your/librispeech/folder
- **file_path**: $S3PRL_DIR/s3prl/data/len_for_bucket
- **urbansound_root**: /path/to/your/urbansound/folder
- **musan_root**: /path/to/your/musan/folder
- **reverb_root**: /path/to/your/impulse_responses_000/folder

You can edit specific details of the model (e.g., which Teacher, use the enhancement head or not, etc.) in the *config_model.yaml* file.

Lastly, to run the pretrain:

```bash
python run_pretrain.py -u robust_distiller -g pretrain/robust_distiller/config_model.yaml -n rd_wavlm
```

## Downstream tasks

This is an example of how to train the speaker diarization (SD) downstream task.
To prepare the dataset and understand more about the task, please look for more specific details [here](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md).

The difference here is that you need to point to where your ckpt file is saved, as in the example:

```bash
python run_downstream.py -n rd_wavlm_cl_sd  -m train -u robust_distiller_local -d diarization -k $S3PRL_DIR/s3prl/result/pretrain/rd_wavlm_cl/states-epoch-17.ckpt
```

# Pretrained Models


In the Hugging Face repository ([Link](https://huggingface.co/Hguimaraes/robustdistiller)), we provide the pretrained weights to the following robust models:

| Base recipe | Teacher | Name on our paper | Link |
|-------------|---------|-------------------|------|
| DistilHuBERT | HuBERT | RD (HuBERT) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller/blob/main/rd_hubert.ckpt) |
| DistilHuBERT | WavLM | RD (WavLM) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller/blob/main/rd_wavlm.ckpt) |
| DistilHuBERT | Robust HuBERT | RD (Robust HuBERT) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller/blob/main/rd_rhubert.ckpt) |
| DPHuBERT | WavLM | RD (DPWavLM) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller) |

# Citation
```latex
@inproceedings{guimaraes2023robustdistiller,
  title={RobustDistiller: Compressing Universal Speech Representations for Enhanced Environment Robustness},
  author={Guimarães, Heitor R and Pimentel, Arthur and Avila, Anderson R and Rezagholizadeh, Mehdi and Chen, Boxing and Falk, Tiago H},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
