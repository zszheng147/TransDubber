import torch
from torch.nn import functional as F


class Token2Wav:
    def __init__(
        self,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
        fp16: bool = False
    ):
        self.device = torch.device("cpu")
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16

    def to(self, device):
        self.device = torch.device(device)
        self.flow = self.flow.to(self.device)
        self.hift = self.hift.to(self.device)
        if self.fp16:
            # Only flow runs in fp16 — it's the iterative ODE that dominates
            # decode latency. HiFi-GAN stays fp32 because its f0_predictor is
            # forced onto CPU (generator.py inference) and CPU conv1d can't
            # mix fp32 input with fp16 weights. flow.inference ends with
            # feat.float() so the fp32 mel hits hift naturally.
            self.flow = self.flow.half()
        return self

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, map_location="cpu", weights_only=True), strict=True)
        self.flow.eval()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.eval()

    def token2wav(self, token, prompt_token, prompt_feat, embedding, speed=1.0,
                  n_timesteps=10, cfg=True,
                  prompt_token_emb=None, embedding_proj=None):
        with torch.amp.autocast("cuda", enabled=self.fp16):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=False,
                finalize=True,
                n_timesteps=n_timesteps,
                cfg=cfg,
                prompt_token_emb=prompt_token_emb,
                embedding_proj=embedding_proj,
            )
            if speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech