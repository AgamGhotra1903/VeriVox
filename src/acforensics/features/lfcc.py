"""Linear Frequency Cepstral Coefficients (LFCC) Extractor with Deltas and Delta-Deltas.

"""

import math
import torch
import torch.nn as nn
import torchaudio.functional as F

from acforensics.constants import SAMPLE_RATE


def create_linear_filterbank(
    n_filters: int = 20,
    n_fft: int = 512,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Constructs a linear triangular filterbank spanning 0 Hz to Nyquist frequency.

    Unlike Mel filterbanks (which compress high frequencies logarithmically),
    linear filterbanks preserve uniform frequency resolution across the full spectrum.
    This is critical for detecting synthetic speech artifacts, which often concentrate
    in higher formant regions and phase transitions.

    Args:
        n_filters: Number of triangular filters (default: 20).
        n_fft: FFT window size (default: 512).
        sample_rate: Sampling frequency in Hz (default: 16000).

    Returns:
        Filterbank matrix of shape (n_filters, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1
    nyquist = sample_rate / 2.0
    all_freqs = torch.linspace(0, nyquist, n_freqs)

    filter_centers = torch.linspace(0, nyquist, n_filters + 2)

    fb = torch.zeros(n_filters, n_freqs, dtype=torch.float32)

    for i in range(n_filters):
        f_left = filter_centers[i]
        f_center = filter_centers[i + 1]
        f_right = filter_centers[i + 2]

        left_mask = (all_freqs >= f_left) & (all_freqs <= f_center)
        if f_center > f_left:
            fb[i, left_mask] = (all_freqs[left_mask] - f_left) / (f_center - f_left)

        right_mask = (all_freqs > f_center) & (all_freqs <= f_right)
        if f_right > f_center:
            fb[i, right_mask] = (f_right - all_freqs[right_mask]) / (f_right - f_center)

    return fb


def create_dct_matrix(n_coeffs: int = 20, n_filters: int = 20) -> torch.Tensor:
    """Generates Type-II Orthonormal Discrete Cosine Transform (DCT) matrix.

    Args:
        n_coeffs: Number of output cepstral coefficients.
        n_filters: Number of input filterbank channels.

    Returns:
        DCT transformation matrix of shape (n_coeffs, n_filters).
    """
    dct_mat = torch.zeros(n_coeffs, n_filters, dtype=torch.float32)
    for k in range(n_coeffs):
        for n in range(n_filters):
            dct_mat[k, n] = math.cos(math.pi * k * (2 * n + 1) / (2 * n_filters))

    dct_mat[0, :] *= 1.0 / math.sqrt(n_filters)
    dct_mat[1:, :] *= math.sqrt(2.0 / n_filters)

    return dct_mat


class LFCCExtractor(nn.Module):
    """Linear Frequency Cepstral Coefficients (LFCC) Feature Extractor.

    Pipeline:
        Raw Waveform (B, T)
        -> Short-Time Fourier Transform (STFT)
        -> Power Spectrum |X|^2
        -> Linear Triangular Filterbank dot product
        -> Log-Compression: log(E + eps)
        -> Orthonormal DCT-II -> 20 Cepstral Coefficients
        -> 1st Delta (Velocity) & 2nd Delta (Acceleration) computation
        -> Stack to 60 total coefficients x ~400 temporal frames
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_filter: int = 20,
        n_lfcc: int = 20,
        with_deltas: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.with_deltas = with_deltas
        self.eps = eps

        fb = create_linear_filterbank(n_filters=n_filter, n_fft=n_fft, sample_rate=sample_rate)
        dct = create_dct_matrix(n_coeffs=n_lfcc, n_filters=n_filter)
        window = torch.hann_window(win_length)

        self.register_buffer("filterbank", fb)
        self.register_buffer("dct_matrix", dct)
        self.register_buffer("window", window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts LFCC features (+ delta + delta-delta).

        Args:
            waveform: Audio tensor of shape (samples,) or (batch_size, samples).

        Returns:
            LFCC tensor of shape (60, frames) or (batch_size, 60, frames).
        """
        is_1d = waveform.dim() == 1
        if is_1d:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        power_spec = stft.abs().pow(2.0)

        fb_energy = torch.einsum("mf, bft -> bmt", self.filterbank, power_spec)

        log_energy = torch.log(fb_energy + self.eps)

        lfcc_static = torch.einsum("km, bmt -> bkt", self.dct_matrix, log_energy)

        if not self.with_deltas:
            return lfcc_static.squeeze(0) if is_1d else lfcc_static

        delta1 = F.compute_deltas(lfcc_static, win_length=5)
        delta2 = F.compute_deltas(delta1, win_length=5)

        lfcc_full = torch.cat([lfcc_static, delta1, delta2], dim=1)

        return lfcc_full.squeeze(0) if is_1d else lfcc_full
