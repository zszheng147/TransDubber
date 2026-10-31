import onnxruntime
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

import whisper

from .spectrogram import mel_spectrogram


class Token2WavFrontEnd:
    def __init__(self, campplus_model: str, speech_tokenizer_model: str):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, 
            providers=["CPUExecutionProvider"]
        )

        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model, sess_options=option,
            providers=["CPUExecutionProvider"]
        )

    def extract_speaker_embedding(self, speech):
        feat = kaldi.fbank(
            speech,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
        })[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding

    def extract_semantic_tokens(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,{
            self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            self.speech_tokenizer_session.get_inputs()[1].name:
            np.array([feat.shape[2]], dtype=np.int32)
        })[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
        return speech_token, speech_token_len

    def extract_mel_features(self, speech):
        # it has been to 24K
        speech_feat = mel_spectrogram(
            speech, n_fft=1920, num_mels=80, 
            sampling_rate=24000, hop_size=480, 
            win_size=1920, fmin=0, fmax=None, center=False
        ).squeeze(dim=0).transpose(0, 1)

        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len