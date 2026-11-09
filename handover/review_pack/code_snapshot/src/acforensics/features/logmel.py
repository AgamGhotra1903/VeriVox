"""Log-Mel Spectrogram Feature Extractor.

Author: Paras Bansal (IIT2024221)
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from acforensics.constants import SAMPLE_RATE


class LogMelExtractor(nn.Module):
    """Computes Log-Mel Spectrogram features.

    Settings:
        Sampling rate: 16,000 Hz
        STFT: n_fft = 512, win_length = 400 (25ms), hop_length = 160 (10ms)
        Mel filterbank: 80 bands (f_min = 0 Hz, f_max = 8,000 Hz)
        Amplitude scaling: Log-compression log(power + 1e-6) or dB conversion.
        Output Shape: (80, ~400) for 4.0s utterance.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        power: float = 2.0,
        top_db: float = 80.0,
        use_db: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self.use_db = use_db
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts Log-Mel Spectrogram.

        Args:
            waveform: Tensor of shape (samples,) or (batch_size, samples).

        Returns:
            Tensor of shape (80, frames) or (batch_size, 80, frames).
        """
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_transform(waveform)

        if self.use_db:
            log_mel = self.amp_to_db(mel_spec)
        else:
            log_mel = torch.log(mel_spec + self.eps)

        return log_mel.squeeze(0) if is_1d else log_mel
