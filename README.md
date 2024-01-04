# RobustDistiller

**tl;dr**: Robust distillation recipe for self-supervised speech representation learning (S3RL) models that tackle jointly model compression and robustness against environmental artifacts (noise and reverberation).

*In this repository, you will find useful codes to train/run the model and links to pre-trained weights.*

# Overview

This repository contain codes and artifacts are from two papers:

1. [ICASSP 2023] RobustDistiller: Compressing Universal Speech Representations for Enhanced Environment Robustness

2. [Journal Extension - Under Review] An Efficient End-to-End Approach to Noise Invariant Speech Features via Multi-Task Learning

**Problem statement**: Self-supervised speech representation learning enables the extraction of meaningful factors from raw waveforms. These features can then be efficiently used across multiple downstream tasks. However, two significant issues arise when considering the deployment of such methods ``in-the-wild": (i) Their large size, which can be prohibitive for edge applications; and (ii) their robustness to detrimental factors, such as noise and/or reverberation, that can heavily degrade the performance of such systems.

**Our proposal**

> We propose a novel knowledge distillation mechanism, namely RobustDistiller, to tackle both problems jointly. 
Two main modifications are proposed to improve the robustness of the student models: (i) a **feature-denoising knowledge distillation** step that induces the student model to learn noise-invariant representations; and (ii) a **multi-task learning approach via a signal enhancement** step where, given the last hidden state from the student model obtained from a noisy signal, we reconstruct the clean waveform or STFT of the clean input.

![alt text](./assets/robustdistiller.png "RobustDistiller")

Initially, in the ICASSP paper, we only evaluated RobustDistiller on 3 downstream tasks. Later, on a journal extension, we evaluate it over 12 downstream tasks and has been shown to outperform several benchmarks regardless of noise type, noise level, and reverberation times. 

Lastly, we show that our recipe is general enough to adapt straightforwardly to other distillation methodologies, such as the recent **DPWavLM**.

# Usage


## Training the upstream model


## Downstream tasks

More details on training for specific downstream tasks can be found [here](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md).

# Pretrained Models


In the Hugging Face repository ([Link](https://huggingface.co/Hguimaraes/robustdistiller)), we provide the pretrained weights to the following robust models:

| Base recipe | Teacher | Name on our paper | Link |
|-------------|---------|-------------------|------|
| DistilHuBERT | HuBERT | RD (HuBERT) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller) |
| DistilHuBERT | WavLM | RD (WavLM) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller) |
| DistilHuBERT | Robust HuBERT | RD (Robust HuBERT) | [CKPT](https://huggingface.co/Hguimaraes/robustdistiller) |
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
