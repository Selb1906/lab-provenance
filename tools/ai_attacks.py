"""AI 공격: 학습된 디노이저(FFDNet)와 재생성(SD-VAE 인코드→디코드).

FFDNet 구조와 가중치는 내 연구 core_AX src/ai_attacks.py 와 assets/models 에서 가져옴
(가중치 출처 KAIR v1.0, Apache-2.0). 회색조 모델이므로 RGB 각 채널에 따로 적용한다.
SD-VAE는 core_AX GEN-3.2와 같은 stabilityai/sd-vae-ft-mse. CPU라 float32로 돌린다.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_DIR = Path("E:/envs/c2pa-work/models")
_CACHE = {}


class _FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_channels=64, n_layers=15):
        super().__init__()
        self.sf = 2
        layers = [nn.Conv2d(in_nc * 4 + 1, n_channels, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(n_channels, out_nc * 4, 3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, sigma):
        _, _, h, w = x.shape
        x_sub = nn.functional.pixel_unshuffle(x, self.sf)
        nm = sigma.view(-1, 1, 1, 1).expand(-1, 1, h // self.sf, w // self.sf)
        return nn.functional.pixel_shuffle(self.model(torch.cat([x_sub, nm], 1)), self.sf)


def _ffdnet():
    if "ffdnet" not in _CACHE:
        m = _FFDNet()
        m.load_state_dict(torch.load(MODEL_DIR / "ffdnet_gray.pth", map_location="cpu", weights_only=True))
        _CACHE["ffdnet"] = m.eval()
    return _CACHE["ffdnet"]


@torch.no_grad()
def ffdnet(img, sigma=25.0):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    ph, pw = h % 2, w % 2
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(1)  # 채널을 배치로
    out = _ffdnet()(x, torch.full((3,), sigma / 255.0))
    out = out.squeeze(1).numpy().transpose(1, 2, 0)[:h, :w]
    return Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8))


def _vae():
    if "vae" not in _CACHE:
        from diffusers import AutoencoderKL
        _CACHE["vae"] = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).eval()
    return _CACHE["vae"]


@torch.no_grad()
def vae_regen(img):
    im = img.convert("RGB")
    w, h = im.size
    W, H = (w // 8) * 8, (h // 8) * 8
    a = np.asarray(im.crop((0, 0, W, H)), dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)
    vae = _vae()
    z = vae.encode(x).latent_dist.mode()
    y = vae.decode(z).sample[0].numpy().transpose(1, 2, 0)
    out = np.asarray(im, dtype=np.float32).copy()
    out[:H, :W] = (y + 1.0) * 127.5
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


ATTACKS = {"ffdnet25": lambda im: ffdnet(im, 25.0), "sdvae": vae_regen}
