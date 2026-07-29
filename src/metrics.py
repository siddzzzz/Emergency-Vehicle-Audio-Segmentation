import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_si_sdr(estimated, target, zero_mean=True, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) computation.
    
    Args:
        estimated: (Batch, Time) or (Time,) reconstructed audio waveform
        target: (Batch, Time) or (Time,) ground-truth audio waveform
    Returns:
        si_sdr_val: Mean SI-SDR value in dB across batch
    """
    if estimated.dim() == 1:
        estimated = estimated.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    if zero_mean:
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

    # Calculate optimal scaling factor alpha
    dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / target_energy

    # Projection and residual
    s_target = alpha * target
    e_noise = estimated - s_target

    target_norm = torch.sum(s_target ** 2, dim=-1) + eps
    noise_norm = torch.sum(e_noise ** 2, dim=-1) + eps

    si_sdr_db = 10.0 * torch.log10(target_norm / noise_norm)
    return torch.mean(si_sdr_db)


class AudioSeparationLoss(nn.Module):
    """
    Combined Loss Function for Multi-Task Audio Source Separation & Vehicle Classification:
    1. L1 Spectral Magnitude Loss (STFT Domain)
    2. SI-SDR Time-Domain Loss
    3. Multi-Class Cross Entropy Loss
    """
    def __init__(self, spectral_weight=1.0, wave_weight=0.05, cls_weight=0.5):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.wave_weight = wave_weight
        self.cls_weight = cls_weight
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, est_mag, target_mag, est_wave=None, target_wave=None, class_logits=None, target_class=None):
        spec_loss = self.l1_loss(est_mag, target_mag)
        total_loss = self.spectral_weight * spec_loss
        
        if class_logits is not None and target_class is not None:
            cls_loss = self.ce_loss(class_logits, target_class)
            total_loss = total_loss + self.cls_weight * cls_loss

        if est_wave is not None and target_wave is not None:
            min_len = min(est_wave.shape[-1], target_wave.shape[-1])
            est_w = est_wave[..., :min_len]
            tgt_w = target_wave[..., :min_len]
            
            sdr_val = calculate_si_sdr(est_w, tgt_w)
            sdr_loss = -torch.clamp(sdr_val, min=-30.0, max=30.0)
            total_loss = total_loss + self.wave_weight * sdr_loss

        return total_loss


if __name__ == "__main__":
    est = torch.randn(4, 16000)
    tgt = est + 0.1 * torch.randn(4, 16000)
    sdr = calculate_si_sdr(est, tgt)
    print("Test SI-SDR (dB):", sdr.item())
