import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):

    def forward(self, x):
        ih, iw = x.shape[-2:]
        (kh, kw), (sh, sw) = self.kernel_size, self.stride
        ph = max((math.ceil(ih / sh) - 1) * sh + kh - ih, 0)
        pw = max((math.ceil(iw / sw) - 1) * sw + kw - iw, 0)
        x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
        return super().forward(x)


class ConvTranspose2dSame(nn.ConvTranspose2d):

    def forward(self, x):
        y = super().forward(x)  # padding=0 -> full output (in-1)*s + k
        ih, iw = x.shape[-2:]
        (kh, kw), (sh, sw) = self.kernel_size, self.stride
        oh, ow = ih * sh, iw * sw
        top = max((ih - 1) * sh + kh - oh, 0) // 2
        left = max((iw - 1) * sw + kw - ow, 0) // 2
        return y[..., top:top + oh, left:left + ow]

class FlattenHWC(nn.Module):

    def forward(self, x):
        return x.permute(0, 2, 3, 1).flatten(1)

class ReshapeHWC(nn.Module):

    def __init__(self, shape_hwc):
        super().__init__()
        self.shape_hwc = tuple(shape_hwc)

    def forward(self, x):
        return x.view(x.size(0), *self.shape_hwc).permute(0, 3, 1, 2).contiguous()

def _bn(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)

def conv_bn_silu(cin, cout, k, s):
    return nn.Sequential(Conv2dSame(cin, cout, k, s), _bn(cout), nn.SiLU())

def deconv_bn_silu(cin, cout, k, s):
    return nn.Sequential(ConvTranspose2dSame(cin, cout, k, s), _bn(cout), nn.SiLU())

class VAE(nn.Module):

    def __init__(self, latent_dim=64, img_shape=(84, 128, 3)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        H, W, C = img_shape

        # two stride-2 convs with 'same' padding -> ceil twice
        self.feat_shape = (math.ceil(math.ceil(H / 2) / 2),
                           math.ceil(math.ceil(W / 2) / 2), 128)
        feat_dim = math.prod(self.feat_shape)

        self.encoder = nn.Sequential(
            conv_bn_silu(C, 32, 3, 2),
            conv_bn_silu(32, 64, 5, 2),
            conv_bn_silu(64, 128, 3, 1),
            FlattenHWC(),
        )
        self.mu = nn.Linear(feat_dim, latent_dim)
        self.logvar = nn.Linear(feat_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feat_dim),
            nn.SiLU(),
            ReshapeHWC(self.feat_shape),
            deconv_bn_silu(128, 128, 3, 2),
            deconv_bn_silu(128, 64, 3, 2),
            deconv_bn_silu(64, 32, 3, 1),
            Conv2dSame(32, 3, 3, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10.0, 10.0)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = self.decoder(z)
        return recon, mu, logvar

    def conv_bn_pairs(self):
        blocks = [*self.encoder[:3], *self.decoder[3:6]]
        return [(b[0], b[1]) for b in blocks] + [(self.decoder[6], None)]