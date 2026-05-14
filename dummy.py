# resave_checkpoint.py
import torch
from dataset import get_dataloaders

# Load your old checkpoint
ckpt = torch.load("checkpoint_train.pt", map_location="cpu")

# Build vocab
_, _, _, assets = get_dataloaders(batch_size=1)

# Add vocab to checkpoint
ckpt["src_vocab"] = assets["src_vocab"]
ckpt["tgt_vocab"] = assets["tgt_vocab"]

# Save back
torch.save(ckpt, "checkpoint_with_vocab.pt")
print("Done! Keys:", ckpt.keys())