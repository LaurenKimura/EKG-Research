"""
model.py
--------
A 1D Convolutional Neural Network for classifying EKG heartbeat segments.

Architecture rationale:
- 1D convolutions slide filters along the time axis of the heartbeat signal,
  which is well suited to detecting local waveform shapes (P wave, QRS
  complex, T wave) regardless of exactly where they occur in the window.
- Stacking conv layers lets the network build from low-level shape detectors
  (edges/slopes) to higher-level pattern detectors (full QRS morphology).
- Global average pooling + a small fully-connected head keeps the parameter
  count low, which helps avoid overfitting on the relatively small number
  of examples in minority arrhythmia classes.
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self, num_classes=5, input_len=180):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # len -> input_len/2

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # len -> input_len/4

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool -> (batch, 64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, input_len) -> add channel dim -> (batch, 1, input_len)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        return self.classifier(x)