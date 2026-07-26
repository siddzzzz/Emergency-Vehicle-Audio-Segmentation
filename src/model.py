import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Dual 2D Convolutional Block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpectrogramUNet(nn.Module):
    """
    Lightweight 2D Spectrogram U-Net for Emergency Vehicle Audio Separation & Source Masking.
    Predicts an Ideal Ratio Mask M in [0, 1] applied to the Mixture Spectrogram.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path
        curr_channels = in_channels
        for feature in features:
            self.encoders.append(ConvBlock(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder Path
        rev_features = list(reversed(features))
        curr_channels = features[-1] * 2
        for feature in rev_features:
            self.decoders.append(
                nn.ConvTranspose2d(curr_channels, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(feature * 2, feature))
            curr_channels = feature

        # Final Ratio Mask Output Head (Sigmoid forces values between 0.0 and 1.0)
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Spectrogram magnitude tensor of shape (Batch, 1, Freq, Time)
        Returns:
            mask: Soft ratio mask of shape (Batch, 1, Freq, Time)
            estimated_siren_mag: Masked Spectrogram (Batch, 1, Freq, Time)
        """
        skip_connections = []

        # Encoder loop
        out = x
        for encoder in self.encoders:
            out = encoder(out)
            skip_connections.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        skip_connections = skip_connections[::-1]

        # Decoder loop with skip connections
        for idx in range(0, len(self.decoders), 2):
            up_sample = self.decoders[idx]
            conv_block = self.decoders[idx + 1]
            
            out = up_sample(out)
            skip = skip_connections[idx // 2]

            # Handle potential shape mismatches due to odd STFT dimension sizes
            if out.shape != skip.shape:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            concat_skip = torch.cat((skip, out), dim=1)
            out = conv_block(concat_skip)

        mask = self.final_conv(out)
        
        # Ensure mask matches input dimensions exactly
        if mask.shape != x.shape:
            mask = F.interpolate(mask, size=x.shape[2:], mode="bilinear", align_corners=False)
            
        estimated_siren_mag = mask * x
        return mask, estimated_siren_mag


if __name__ == "__main__":
    model = SpectrogramUNet()
    # Test batch with shape (Batch=2, Channel=1, Freq=257, Time=401)
    dummy_input = torch.randn(2, 1, 257, 401)
    mask, est_mag = model(dummy_input)
    print("Model Input Shape:", dummy_input.shape)
    print("Predicted Mask Shape:", mask.shape)
    print("Estimated Siren Mag Shape:", est_mag.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params:,}")
