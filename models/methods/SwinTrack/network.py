import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))



class SwinTrack(nn.Module):
    def __init__(self, backbone, encoder, decoder, out_norm, head,
                 z_backbone_out_stage, x_backbone_out_stage,
                 z_input_projection, x_input_projection,
                 z_pos_enc, x_pos_enc):
        super(SwinTrack, self).__init__()
        self.device = 'cuda'
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.out_norm = out_norm
        self.head = head

        self.z_backbone_out_stage = z_backbone_out_stage
        self.x_backbone_out_stage = x_backbone_out_stage
        self.z_input_projection = z_input_projection
        self.x_input_projection = x_input_projection

        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc

        d_model = 512 
        ratio = 2
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * ratio),
            nn.GELU(),
            nn.Linear(d_model * ratio, d_model),
        )
        self.cond_prj = nn.Sequential(
            nn.Linear(d_model+4, d_model * ratio),
            nn.GELU(),
            nn.Linear(d_model * ratio, d_model),
        )

        self.reset_parameters()


        # build diffusion
        timesteps = 300   #训练时的采样步数
        sampling_timesteps = 20 #cfg.MODEL.DiffusionDet.SAMPLE_STEP # test时的步数，增加 （>50, 100, 200） 2卡
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = sampling_timesteps #default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2. #cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.z_input_projection is not None:
            self.z_input_projection.apply(_init_weights)
        if self.x_input_projection is not None:
            self.x_input_projection.apply(_init_weights)

        self.encoder.apply(_init_weights)
        # self.decoder.apply(_init_weights)

    def initialize(self, z):
        return self._get_template_feat(z)

    def track(self, z_feat, x, gt_bbox=None):
        x_feat = self._get_search_feat(x)

        return self._track(z_feat, x_feat, gt_bbox)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_targets(self, targets):
        t = torch.randint(0, self.num_timesteps, (targets.shape[0],), device=self.device).long()
        noise = torch.randn(targets.shape, dtype=torch.float32, device=self.device)

        x_start = targets
        x_start = (x_start * 2. - 1.) * self.scale

        x = self.q_sample(x_start=x_start, t=t, noise=noise).float()

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        return x, noise, t

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, z_feat, x_feat, z_pos, x_pos, x, t):
        # return self.head(self.decoder(backbone_feats+x,t))
        targets = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        targets = ((targets / self.scale) + 1) / 2

        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1)

        results = self.decoder(z_feat, x_feat, z_pos, x_pos, targets, t_emb)

        x_start =torch.concat([results['bbox'], results['class_score'].permute(0, 2, 3, 1)], dim=-1)

        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        x_start = x_start.reshape(x.shape)
        pred_noise = self.predict_noise_from_start(x, t, x_start).float()

        return pred_noise, x_start, results

    @torch.no_grad()
    def ddim_sample(self, z_feat, x_feat, z_pos, x_pos):
        batch, l = x_feat.shape[:2]
        shape = (batch, l, 5)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, dtype=torch.float, device=self.device)

        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            pred_noise, x_start, results = self.model_predictions(z_feat, x_feat, z_pos, x_pos, img, time_cond) # ?

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return results 

    def forward(self, z, x, z_feat=None, gt_bbox=None):
        """
        Combined entry point for training and inference (include initialization and tracking).
            Args:
                z (torch.Tensor | None)
                x (torch.Tensor | None)
                z_feat (torch.Tensor | None)
                gt_bbox (torch.Tensor | None)   here gt_bbox has shape (H*W, 5) and the last dim is classification result
            Training:
                Input:
                    z: (B, H_z * W_z, 3), template image
                    x: (B, H_x * W_x, 3), search image
                Return:
                    Dict: Output of the head, like {'class_score': torch.Tensor(B, num_classes, H, W), 'bbox': torch.Tensor(B, H, W, 4)}.
            Inference:
                Initialization:
                    Input:
                        z: (B, H_z * W_z, 3)
                    Return:
                        torch.Tensor: (B, H_z * W_z, dim)
                Tracking:
                    Input:
                        z_feat: (B, H_z * W_z, dim)
                        x: (B, H_x * W_x, 3)
                    Return:
                        Dict: Same as training.
            """

        if z_feat is None:
            z_feat = self.initialize(z)
        if x is not None:
            return self.track(z_feat, x, gt_bbox)
        else:
            return z_feat

    def _get_template_feat(self, z):
        z_feat, = self.backbone(z, (self.z_backbone_out_stage,), False)
        if self.z_input_projection is not None:
            z_feat = self.z_input_projection(z_feat)
        return z_feat

    def _get_search_feat(self, x):
        x_feat, = self.backbone(x, (self.x_backbone_out_stage,), False)
        if self.x_input_projection is not None:
            x_feat = self.x_input_projection(x_feat)
        return x_feat
    
    def _track(self, z_feat, x_feat, gt_bbox=None):
        z_pos = None
        x_pos = None

        if self.z_pos_enc is not None:
            z_pos = self.z_pos_enc().unsqueeze(0)
        if self.x_pos_enc is not None:
            x_pos = self.x_pos_enc().unsqueeze(0)

        z_feat, x_feat = self.encoder(z_feat, x_feat, z_pos, x_pos)

        if gt_bbox == None:
            gt_bbox = x_feat[:,:,:5]

        # replace orignal decoder
        if not self.training:
            results = self.ddim_sample(z_feat, x_feat, z_pos, x_pos)
            return results
        else:
            targets, noises, t = self.prepare_targets(gt_bbox)

            t = self.time_mlp(t)
            t = t.unsqueeze(1)

            results = self.decoder(z_feat, x_feat, z_pos, x_pos, targets, t)
            return results
