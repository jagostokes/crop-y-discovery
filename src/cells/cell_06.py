"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 6).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Sequence-aware augmentation
# ---------------------------------------------------------------------------
# torchvision.transforms operate on (C, H, W) tensors. Our images are
# (T, C, H, W). We can't just call RandomHorizontalFlip — it would flip
# each frame independently, which would scramble the temporal sequence.
#
# Solution: write a tiny wrapper that decides ONCE per sequence whether
# to flip, then applies that decision to every frame. This preserves the
# temporal coherence of the augmented sequence.
# ---------------------------------------------------------------------------
import torch.nn.functional as F

class SequenceFlipAugment:
    """Random h/v flips applied identically to all frames in a sequence."""

    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img_seq):
        # img_seq: (T, C, H, W)
        if torch.rand(1).item() < self.p_hflip:
            img_seq = torch.flip(img_seq, dims=[-1])  # flip width
        if torch.rand(1).item() < self.p_vflip:
            img_seq = torch.flip(img_seq, dims=[-2])  # flip height
        return img_seq

train_augment = SequenceFlipAugment(p_hflip=0.5, p_vflip=0.5)
