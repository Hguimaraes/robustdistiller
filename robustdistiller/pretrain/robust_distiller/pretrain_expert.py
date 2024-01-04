from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretrain.robust_distiller.dataset import OnlineWaveDataset
from upstream.robust_distiller.model import DistillerConfig, DistillerModel
from .robustness import RobustEnvironment
from .enhancement_losses import MultiResolutionSTFTLoss

from upstream.hubert.hubconf import hubert
from upstream.hubert.hubconf import hubert_base_robust_mgr
from upstream.wav2vec2.hubconf import wav2vec2
from upstream.wavlm.hubconf import wavlm

import numpy as np
from pesq import pesq_batch
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = DistillerConfig(self.upstream_config["distiller"])
        self.model = DistillerForPretrain(
            model_config, 
            edict(self.upstream_config["teacher"]), 
            self.datarc["robust_env"],
            self.upstream_config["enhancement_head"]
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model(
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value
            print(f'[Runner] Logs : {records}')

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)


class DistillerForPretrain(nn.Module):
    """
    Distiller for pretraining
    """

    def __init__(
        self,
        config: DistillerConfig,
        teacher_config: edict,
        robust_env_config: dict,
        enhancement_head: dict,
    ):
        super().__init__()
        self.config = config
        self.distiller = DistillerModel(config)
        self.enhancement_task = enhancement_head['use_enhancement']
        self.lambda_enhancement = enhancement_head['lambda']
        self.enhancement_loss = robust_env_config['enhancement_loss']
        
        if self.enhancement_loss == "mrstft":
            self.enhc_loss = MultiResolutionSTFTLoss()

        if self.enhancement_loss == "si_snr":
            from asteroid.losses import PITLossWrapper
            from asteroid.losses.sdr import SingleSrcNegSDR
            self.enhc_loss = PITLossWrapper(
                SingleSrcNegSDR("sisdr"),
                pit_from='pw_pt'
            )

        if self.enhancement_loss == "l1":
            self.enhc_loss = F.l1_loss

        if self.enhancement_loss == "l2":
            self.enhc_loss = F.mse_loss

        hann_window = torch.hann_window(400).to("cuda")
        self.stft = lambda x: torch.stft(
            x,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=hann_window,
            center=True,
            return_complex=True
        )

        self.inv_stft = lambda x, length: torch.istft(
            x,
            length=length,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=hann_window,
            center=True
        )

        if self.enhancement_task:
            self.reconstruction_head = ReconstructionHead(
                input_dim=enhancement_head['input_dim'],
                num_bins=enhancement_head['num_bins'],
                num_layers=enhancement_head['num_layers'],
                hidden_size=enhancement_head['hidden_size'],
                dropout=enhancement_head['dropout'],
                bidirectional=enhancement_head['bidirectional']
            )

        self.teacher_config = teacher_config
        fn_method = {
            "hubert": hubert(),
            "wav2vec2": wav2vec2(),
            "wavlm": wavlm(),
            "robust_hubert": hubert_base_robust_mgr()
        }
        teacher = fn_method[self.teacher_config.model]
        teacher.model.encoder.layerdrop = 0

        assert self.distiller.n_tasks <= teacher_config.n_layers, (
            self.distiller.n_tasks,
            teacher_config.n_layers,
        )
        self.teacher = teacher
        freeze_model(self.teacher)

        print(
            "[DistillerForPretrain] - Using {} as teacher with {} layers".format(
                teacher_config.model, teacher_config.n_layers
            )
        )

        if config.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="none")
        elif config.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(config.loss_type)

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        if config.init_teacher_conv_layers:
            print(
                "[DistillerForPretrain] - "
                "Initializing feature extractor from teacher"
            )
            self.distiller.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict()
            )
            if self.distiller.post_extract_proj is not None:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            print("[DistillerForPretrain] - " "Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict()
            )

            for l in range(config.encoder_layers):
                teacher_dict = self.teacher.model.encoder.layers[l].state_dict()

                if teacher_config.model == "wavlm":
                    if l == 0:
                        del teacher_dict['self_attn.relative_attention_bias.weight']

                    del teacher_dict['self_attn.grep_a']
                    del teacher_dict["self_attn.grep_linear.weight"]
                    del teacher_dict["self_attn.grep_linear.bias"]

                self.distiller.encoder.layers[l].load_state_dict(teacher_dict)

        self.robust_env = RobustEnvironment(robust_env_config)

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """
        # Corrupt the dataset
        noisy_input = self.robust_env.transform(wave_input)

        # Forward model
        feat, feat_final, pred, pad_mask = self.distiller(noisy_input, pad_mask)

        # Forward to the reconstruction head
        rec_stft, n_stft_phase = None, None
        c_stft_mag = None
        if self.enhancement_task:
            feat_upsampled = torch.repeat_interleave(feat, 2, dim=1)
            c_stft_features = self.stft(wave_input)
            c_stft_mag = torch.abs(c_stft_features)

            n_stft_features = self.stft(noisy_input)
            n_stft_mag, n_stft_phase = torch.abs(n_stft_features), torch.angle(n_stft_features)

            feat_upsampled = torch.unbind(feat_upsampled, dim=0)
            stft_lens = [sf.shape[-1] for sf in torch.unbind(c_stft_mag, dim=0)]
            feat_upsampled = match_length(feat_upsampled, stft_lens)
            feat_upsampled = torch.stack(feat_upsampled, dim=0)

            mask_features = self.reconstruction_head(feat_upsampled)
            rec_stft = torch.mul(mask_features.transpose(1, 2), n_stft_mag)

        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                teacher_hiddens = self.teacher(wave_orig)
            if self.config.task_emb_type == "none":
                teacher_hiddens = teacher_hiddens["hidden_states"][self.config.n_tasks]
                teacher_hiddens = teacher_hiddens.unsqueeze(1)
            else:
                if self.config.task_emb_type in ["expand-last", "hnet", "self-hidden"]:
                    teacher_hiddens = [
                        teacher_hiddens["hidden_states"][i]
                        for i in self.distiller.pred_layer_id
                    ]
                else:
                    teacher_hiddens = teacher_hiddens["hidden_states"][1:]
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D

        # Compute all objectives
        (
            total_loss,
            rec_loss,
            enh_loss,
            rec_layer_loss,
            feat_pen,
            sim_loss,
            sim_layer_loss,
        ) = self.compute_loss(
            feat,
            pred,
            teacher_hiddens,
            wave_input,
            c_stft_mag,
            rec_stft,
            n_stft_phase,
            return_other
        )

        if return_other:
            with torch.no_grad():
                stoi_score, si_snr_score, pesq_score = 0, 0, 0
                if self.enhancement_task:
                    rec_stft = torch.polar(abs=rec_stft, angle=n_stft_phase)
                    rec_waveform = self.inv_stft(rec_stft, length=wave_input.shape[-1])
                    pesq_score = np.array(pesq_batch(
                        16000,
                        wave_input.detach().cpu().numpy(),
                        rec_waveform.detach().cpu().numpy(),
                        'wb'
                    )).mean()
                    stoi_score = short_time_objective_intelligibility(
                        preds=rec_waveform,
                        target=wave_input,
                        fs=16000
                    ).mean()

                    si_snr_score = scale_invariant_signal_noise_ratio(
                        preds=rec_waveform,
                        target=wave_input
                    ).mean()

                other_res = {
                    "rec_loss": rec_loss,
                    "feat_pen": feat_pen,
                    "enhc_loss": enh_loss,
                    "enhc_stoi":stoi_score,
                    "enhc_si_snr": si_snr_score,
                    "enhc_pesq": pesq_score,
                    "sim_loss": sim_loss,
                    "norm_feat_final": feat_final.pow(2).mean(),
                }
                teacher_norm = torch.abs(teacher_hiddens).mean((0, 2, 3))
                if self.config.task_emb_type == "none":
                    other_res[f"rec_l{self.config.n_tasks}"] = rec_layer_loss[0]
                    other_res[f"tar_norm_l{self.config.n_tasks}"] = teacher_norm[0]
                    if sim_layer_loss is not None:
                        other_res[f"sim_l{self.config.n_tasks}"] = sim_layer_loss[0]
                else:
                    for i in range(self.config.n_tasks):
                        layer_id = i + 1
                        if self.config.task_emb_type in [
                            "expand-last",
                            "hnet",
                            "self-hidden",
                        ]:
                            layer_id = self.distiller.pred_layer_id[i]
                        other_res[f"rec_l{layer_id}"] = rec_layer_loss[i]
                        other_res[f"tar_norm_l{layer_id}"] = teacher_norm[i]
                        if sim_layer_loss is not None:
                            other_res[f"sim_l{layer_id}"] = sim_layer_loss[i]
                    if self.config.task_emb_type not in [
                        "expand-last",
                        "hnet",
                        "self-hidden",
                    ]:
                        other_res[
                            "norm_task_emb"
                        ] = self.distiller.task_embedding.weight.pow(2).mean()
        else:
            other_res = None

        return total_loss, other_res

    def compute_loss(self, feat, pred, target, wave_input, spect, spect_hat, n_stft_phase, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D
        """
        # Filterbank/Waveform reconstruction
        enh_loss = 0
        if self.enhancement_task:
            if (self.enhancement_loss == "mrstft") or (self.enhancement_loss == "si_snr"):
                spect_hat = torch.polar(spect_hat, angle=n_stft_phase)
                rec_waveform = self.inv_stft(spect_hat, length=wave_input.shape[-1])
                if self.enhancement_loss == "si_snr":
                    rec_waveform = rec_waveform.unsqueeze(1)
                    wave_input = wave_input.unsqueeze(1)

                enh_loss = self.enhc_loss(rec_waveform, wave_input)

            else:
                enh_loss = self.enhc_loss(spect_hat, spect)

        # Reconstruction loss
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D

        if return_other:
            with torch.no_grad():
                rec_layer_loss = rec_loss.mean((0, 2, 3))
        else:
            rec_layer_loss = None

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1))
            # B x N x T
            if return_other:
                with torch.no_grad():
                    sim_layer_loss = sim_loss.mean((0, 2))
            else:
                sim_layer_loss = None
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # Feature loss
        feat_pen = feat.float().pow(2).mean()

        total_loss = (
            rec_loss
            + feat_pen * self.config.feat_pen_loss
            + sim_loss * self.cosine_loss
            + enh_loss * self.lambda_enhancement
        )

        return total_loss, rec_loss, enh_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss


def match_length(feat_list, length_list):
    assert len(feat_list) == len(length_list)
    bs = len(length_list)
    new_feat_list = []
    for i in range(bs):
        assert abs(feat_list[i].size(0) - length_list[i]) < 5
        if feat_list[i].size(0) == length_list[i]:
            new_feat_list.append(feat_list[i])
        elif feat_list[i].size(0) > length_list[i]:
            new_feat_list.append(feat_list[i][:length_list[i], :])
        else:
            new_feat = torch.zeros(length_list[i], feat_list[i].size(1)).to(feat_list[i].device)
            new_feat[:feat_list[i].size(0), :] = feat_list[i]
            new_feat_list.append(new_feat)
    return new_feat_list

class ReconstructionHead(nn.Module):
    def __init__(
        self,
        input_dim=768,
        num_bins=257,
        num_layers=3,
        hidden_size=256,
        dropout=0.1,
        bidirectional=True
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.drops = torch.nn.Dropout(p=0.1)
        self.linear = torch.nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_bins
        )

        self.non_linear = torch.sigmoid
        self.num_bins = num_bins

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x, _ = self.rnn(x)
        x = self.drops(x)
        x = self.linear(x)
        return self.non_linear(x)
