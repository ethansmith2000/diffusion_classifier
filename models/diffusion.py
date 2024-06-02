
import torch
from torch import nn
import diffusers
import math

class FourierEmbedder:
    def __init__(self, num_freqs, temperature):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        """
        :param x: arbitrary shape of tensor
        :param cat_dim: cat dim
        """
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class AdaNorm(torch.nn.Module):

    def __init__(self, dim, embed_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(embed_dim, dim * 2, bias=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(self.silu(emb)).chunk(2, dim=-1)
        return x * (scale + 1) + shift


class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        ada_dim: int = 512,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = AdaNorm(in_dim, ada_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = AdaNorm(mid_dim, ada_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()


    def forward(
        self,
        hidden_states,
        ada_emb=None,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states, ada_emb)))
        hidden_states = self.linear2(self.dropout(self.act(self.norm2(hidden_states, ada_emb))))

        return hidden_states + resid


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    downscale_freq_shift: float = 1,
    max_period: int = 10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)


    def forward(self, timestep):
        timestep = get_timestep_embedding(timestep, self.linear_1.in_features).to(self.linear_1.weight.device).to(self.linear_1.weight.dtype)
        timestep = self.linear_1(timestep)
        timestep = self.act(timestep)
        timestep = self.linear_2(timestep)
        return timestep



class DiffusionClassifier(nn.Module):
    def __init__(self, encoder, dim, timestep_dim=64, num_classes=10, num_layers=6):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        self.encoder = encoder
        self.scheduler = diffusers.DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", rescale_betas_zero_snr=True, subfolder="scheduler")
        self.fourier_embedder = FourierEmbedder(num_freqs=32, temperature=100)
        self.proj_in = nn.Linear(32 * 2 * num_classes + timestep_dim, dim)
        self.proj_cond = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, num_classes, bias=False)
        self.time_embed = TimestepEmbedding(dim, timestep_dim)
        self.decoder = nn.ModuleList([Resnet(dim, mid_dim=dim * 2) for _ in range(num_layers)])


    def encode_context(self, image):
        context = self.encoder.get_feat(image)
        context = self.proj_cond(context)
        return context
    
    
    def forward(self, x, context, t):
        t = self.time_embed(t)
        x = self.fourier_embedder(x)
        x = self.proj_in(torch.cat([x, t], dim=-1))
        for layer in self.decoder:
            x = layer(x, context)
        x = self.norm_out(x)
        x = self.proj_out(x)
        return x

    def forward_with_cfg(self, x, context, t, cfg=1.5):
        context = torch.cat([torch.zeros_like(context), context], dim=0)
        ts = torch.cat([t, t], dim=0)
        pred_uncond, pred_cond = self.forward(torch.cat([x,x],dim=0), context, ts).chunk(2)
        return pred_uncond + cfg * (pred_cond - pred_uncond)
        
    def generate(self, images, num_steps=20, cfg=1.0):
        self.scheduler.set_timesteps(num_steps)
        x = torch.randn(images.shape[0], self.num_classes).to(images.device)
        context = self.encode_context(images)
        for i, t in enumerate(self.scheduler.timesteps):
            pred = self.forward_with_cfg(x, context, t.unsqueeze(0).expand(context.shape[0]), cfg)
            x = self.scheduler.step(pred, t, x, return_dict=False)[0]

        return x