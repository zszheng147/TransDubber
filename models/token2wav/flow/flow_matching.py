import sys
sys.path.append("..")
from abc import ABC

import torch
import torch.nn.functional as F

from ..utils.common import set_all_random_seed


class BASECFM(torch.nn.Module, ABC):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = getattr(cfm_params, 'sigma_min', 1e-4)
        self.estimator = None


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False, cfg=True):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []
        if cfg:
            x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
            mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
            mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
            t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
            spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
            cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
            for step in range(1, len(t_span)):
                x_in[:] = x
                mask_in[:] = mask
                mu_in[0] = mu
                t_in[:] = t.unsqueeze(0)
                spks_in[0] = spks
                cond_in[0] = cond
                dphi_dt = self.forward_estimator(
                    x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming
                )
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
                x = x + dt * dphi_dt
                t = t + dt
                sol.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t
        else:
            # batch=1, no CFG: skip the doubled-batch trick entirely. `t` was
            # already unsqueezed to shape [1] above, which matches the batch
            # dim expected by DiT.forward.
            for step in range(1, len(t_span)):
                t_in = t.to(spks.dtype)
                dphi_dt = self.forward_estimator(
                    x, mask, mu, t_in, spks, cond, streaming
                )
                x = x + dt * dphi_dt
                t = t + dt
                sol.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t
        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', (2, 80, x.size(2)))
                estimator.set_input_shape('mask', (2, 1, x.size(2)))
                estimator.set_input_shape('mu', (2, 80, x.size(2)))
                estimator.set_input_shape('t', (2,))
                estimator.set_input_shape('spks', (2, 80))
                estimator.set_input_shape('cond', (2, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)
        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False, cfg=True):
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming, cfg=cfg), None
