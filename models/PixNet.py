"""models/PixNet.py -- Kien truc mo hinh PixNet (UNI2-h backbone + decoder SAFB/DSUB
+ low-rank gene head). Chuyen NGUYEN VEN tu pixnet.ipynb (Muc 3: DINH NGHIA MO HINH),
khong doi logic -- chi tach thanh module rieng va them import can thiet.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def make_gn(channels, max_groups=32):
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)

class DSUB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.d2s = nn.PixelShuffle(upscale_factor=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.cb = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_gn(out_channels),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, F_l):
        x = self.conv1(F_l)
        x = self.relu1(x)
        x = self.d2s(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return self.cb(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            make_gn(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.down(x)

class BilinearUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cb1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_gn(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.cb2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_gn(out_channels),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.cb1(x)
        x = self.upsample(x)
        return self.cb2(x)

class SAFB(nn.Module):
    def __init__(self, c_lateral, c_up, c_out):
        super().__init__()
        self.dwc = nn.Conv2d(c_lateral, c_lateral, kernel_size=3, padding=1, groups=c_lateral)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv1x1_mid = nn.Conv2d(c_lateral, c_lateral, kernel_size=1)
        self.ln = nn.GroupNorm(1, c_lateral)
        self.silu2 = nn.SiLU(inplace=True)
        self.bn = make_gn(c_lateral)

        c_concat = c_lateral + c_up
        self.qkv_conv = nn.Conv2d(c_concat, c_concat * 3, kernel_size=1)
        self.out_conv = nn.Conv2d(c_concat, c_out, kernel_size=1)
        self.beta = math.sqrt(c_concat)
        self.res_proj = nn.Identity() if c_concat == c_out else nn.Conv2d(c_concat, c_out, kernel_size=1)
        self.out_norm = make_gn(c_out)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, F_lateral, U):
        dwc_f = self.dwc(F_lateral)
        f_mid = self.silu1(dwc_f)
        f_mid = self.conv1x1_mid(f_mid)
        f_mid = self.ln(f_mid)
        f_hat = dwc_f + f_mid
        f_hat = self.silu2(f_hat)
        f_hat = self.bn(f_hat)

        F_u = torch.cat([f_hat, U], dim=1)
        B, C, H, W = F_u.shape

        qkv = self.qkv_conv(F_u)
        Q, K, V = torch.chunk(qkv, 3, dim=1)
        Q = Q.flatten(2).transpose(1, 2)
        K = K.flatten(2).transpose(1, 2)
        V = V.flatten(2).transpose(1, 2)

        with torch.autocast(device_type='cuda', enabled=False):
            Qf, Kf, Vf = Q.float(), K.float(), V.float()
            logits = (Qf @ Kf.transpose(-2, -1)) / self.beta
            logits = torch.clamp(logits, min=-30.0, max=30.0)
            attn = torch.softmax(logits, dim=-1)
            out = attn @ Vf

        out = out.to(F_u.dtype)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        D = self.res_proj(F_u) + self.out_conv(out)
        return self.out_norm(D)

class LowRankGeneHead(nn.Module):
    def __init__(self, in_channels, K, M):
        super().__init__()
        self.spatial_proj = nn.Conv2d(in_channels, K, kernel_size=1)  # U (có bias)
        self.gene_loading = nn.Parameter(torch.empty(K, M))           # V (không bias)
        nn.init.trunc_normal_(self.spatial_proj.weight, std=0.02)
        nn.init.zeros_(self.spatial_proj.bias)
        nn.init.trunc_normal_(self.gene_loading, std=0.01 / math.sqrt(K))

    def forward(self, D1):
        U = self.spatial_proj(D1)                                    # [B,K,H',W']
        G = torch.einsum('bkhw,km->bmhw', U, self.gene_loading)      # [B,M,H',W']
        return F.softplus(G)                                         # rho >= 0

class PixNet(nn.Module):
    def __init__(self, num_genes=250, use_lowrank=False, lowrank_k=64):
        super().__init__()
        timm_kwargs = {
            'img_size': 224, 'patch_size': 14, 'depth': 24,
            'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2, 'num_classes': 0,
            'no_embed_class': True, 'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU, 'reg_tokens': 8,
            'dynamic_img_size': True,
        }
        self.encoder = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.encoder.eval()
        for param in self.encoder.parameters(): param.requires_grad = False

        n_forced = 0
        for blk in self.encoder.blocks:
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'fused_attn'):
                blk.attn.fused_attn = True
                n_forced += 1

        self._sdpa_ctx = None
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            self._sdpa_backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            self._sdpa_kernel_fn = sdpa_kernel
        except ImportError:
            self._sdpa_kernel_fn = None

        self.embed_dim = C = 1536
        self.extracted_features = {}

        def get_features(name):
            def hook(module, inp, out): self.extracted_features[name] = out
            return hook

        self.encoder.blocks[5].register_forward_hook(get_features('F1'))
        self.encoder.blocks[11].register_forward_hook(get_features('F2'))
        self.encoder.blocks[17].register_forward_hook(get_features('F3'))
        self.encoder.blocks[23].register_forward_hook(get_features('F4_bottom'))

        self.down_F2 = DownsampleBlock(C, C)
        self.down_F3 = nn.Sequential(DownsampleBlock(C, C), DownsampleBlock(C, C))
        self.down_F4 = nn.Sequential(DownsampleBlock(C, C), DownsampleBlock(C, C), DownsampleBlock(C, C))

        self.up_blocks = nn.ModuleList([DSUB(C, 512), BilinearUpBlock(512, 512), BilinearUpBlock(512, 256)])
        self.safbs = nn.ModuleList([SAFB(C, 512, 512), SAFB(C, 512, 512), SAFB(C, 256, 256)])

        self.final_conv = nn.Conv2d(256, num_genes, kernel_size=1)
        nn.init.trunc_normal_(self.final_conv.weight, std=0.01)
        nn.init.zeros_(self.final_conv.bias)

        self.use_lowrank = use_lowrank
        if self.use_lowrank:
            self.lowrank_head = LowRankGeneHead(256, lowrank_k, num_genes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _tokens_to_map(self, tokens):
        B, N_total, C = tokens.shape
        n_prefix = getattr(self.encoder, 'num_prefix_tokens', 1)
        patch_tokens = tokens[:, n_prefix:, :]
        n_patches = patch_tokens.shape[1]
        side = int(round(math.sqrt(n_patches)))
        return patch_tokens.transpose(1, 2).reshape(B, C, side, side)

    def _run_encoder(self, x):
        if self._sdpa_kernel_fn is not None:
            try:
                with self._sdpa_kernel_fn(self._sdpa_backends):
                    return self.encoder.forward_features(x)
            except Exception as e:
                return self.encoder.forward_features(x)
        return self.encoder.forward_features(x)

    def forward(self, x):
        self.extracted_features = {}
        with torch.no_grad(): _ = self._run_encoder(x)

        F1 = self._tokens_to_map(self.extracted_features['F1'])
        F2 = self._tokens_to_map(self.extracted_features['F2'])
        F3 = self._tokens_to_map(self.extracted_features['F3'])
        F4 = self._tokens_to_map(self.extracted_features['F4_bottom'])

        F2 = self.down_F2(F2)
        F3 = self.down_F3(F3)
        F4 = self.down_F4(F4)

        D = self.up_blocks[0](F4)
        D = self.safbs[0](F3, D)
        D = self.up_blocks[1](D)
        D = self.safbs[1](F2, D)
        D = self.up_blocks[2](D)
        D = self.safbs[2](F1, D)

        if self.use_lowrank:
            return self.lowrank_head(D)
        return F.softplus(self.final_conv(D))