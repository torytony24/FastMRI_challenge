import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from unet import Unet
from utils.common.utils import center_crop
from utils.model.aspin_varnet import AnatomyClassifier, SensitivityModel


class SoftASPIN(nn.Module):
    """
    Soft-ASPIN: used for inference
    """
    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, classifier: nn.Module):
        super().__init__()
        self.register_buffer('gamma', gamma)
        self.register_buffer('beta', beta)
        self.classifier = classifier
        self.eps = 1e-5

    def forward(self, x):
        B, C = x.shape[:2]
        probs = self.classifier(x)

        gamma_soft = torch.matmul(probs, self.gamma)
        beta_soft  = torch.matmul(probs, self.beta)

        broadcast_shape = (B, C) + (1,) * (x.ndim - 2)
        gamma_soft = gamma_soft.view(broadcast_shape)
        beta_soft  = beta_soft.view(broadcast_shape)

        dims = tuple(range(2, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()
        norm = (x - mean) / std

        return gamma_soft * norm + beta_soft, mean, std


class SoftASPIN_Unet(nn.Module):
    """
    Soft-ASPIN adapted U-Net
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        gamma, beta, classifier,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

        self.aspin = SoftASPIN(
            gamma = gamma,
            beta = beta,
            classifier = classifier
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    # Not used
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # ASPIN here
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.aspin(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # unnorm
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x
    

class SoftASPIN_VarNetBlock(nn.Module):
    """
    Soft-ASPIN VarNetBlock
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term
    

class SoftASPIN_VarNet(nn.Module):
    """
    Soft-ASPIN VarNet
    """
    def __init__(
        self,
        aspin_varnet,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.gamma = aspin_varnet.gamma.detach().clone()
        self.beta = aspin_varnet.beta.detach().clone()
        self.classifier = aspin_varnet.anatomy_classifier
        self.classifier.eval()
        self.cascades = nn.ModuleList(
            [SoftASPIN_VarNetBlock(SoftASPIN_Unet(chans, pools, gamma = self.gamma, beta = self.beta, classifier = self.classifier)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        result = center_crop(result, 384, 384)
        return result
    
    

