"""
Phase 2: Train a linear projection layer to map
Finetuned BiomedCLIP text hidden states → SD CLIP text hidden states.

Uses paired embeddings from Phase 1 (extract_paired_embeddings.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import os
from tqdm import tqdm


class EmbeddingProjector(nn.Module):
    """Linear projection: BiomedCLIP (768) → SD CLIP (768)."""

    def __init__(self, hidden_dim=768):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, 768)
        return self.proj(x)


class PairedEmbeddingDataset(Dataset):
    """Dataset of paired (BiomedCLIP, SD CLIP) embeddings."""

    def __init__(self, bio_embs, sd_embs):
        self.bio = bio_embs  # (N, 77, 768)
        self.sd = sd_embs  # (N, 77, 768)

    def __len__(self):
        return self.bio.shape[0]

    def __getitem__(self, idx):
        return self.bio[idx], self.sd[idx]


def compute_shift_loss(projector, bio_embs, sd_embs, device):
    """
    Compute shift direction preservation loss using medical anchors.
    The key idea: proj(bio_healthy) - proj(bio_tumor) ≈ sd_healthy - sd_tumor
    """
    # Use the last few pairs which are the breast anchors:
    # "tumor breast" and "healthy breast" are always in the dataset
    # We compute shift for random pairs within the batch
    n = bio_embs.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=device)

    # Random pairs
    idx1 = torch.randperm(n, device=device)[: n // 2]
    idx2 = torch.randperm(n, device=device)[: n // 2]

    bio_shift = projector(bio_embs[idx1]) - projector(bio_embs[idx2])
    sd_shift = sd_embs[idx1] - sd_embs[idx2]

    return F.mse_loss(bio_shift, sd_shift)


def train(args):
    device = args.device

    # Load paired embeddings
    print(f"Loading paired embeddings from: {args.data}")
    data = torch.load(args.data, map_location="cpu")
    bio_embs = data["bio_embeddings"]  # (N, 77, 768)
    sd_embs = data["sd_embeddings"]  # (N, 77, 768)
    print(f"Loaded {bio_embs.shape[0]} pairs, shape: {bio_embs.shape}")

    # Train/Val split
    dataset = PairedEmbeddingDataset(bio_embs, sd_embs)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    projector = EmbeddingProjector(768).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        projector.train()
        train_losses = {"mse": 0, "cos": 0, "shift": 0, "total": 0}
        n_batches = 0

        for bio_batch, sd_batch in train_loader:
            bio_batch = bio_batch.to(device)
            sd_batch = sd_batch.to(device)

            projected = projector(bio_batch)

            # MSE Loss
            loss_mse = F.mse_loss(projected, sd_batch)

            # Cosine Similarity Loss (per-token, averaged)
            cos_sim = F.cosine_similarity(projected, sd_batch, dim=-1)  # (batch, 77)
            loss_cos = 1 - cos_sim.mean()

            # Shift Loss
            loss_shift = compute_shift_loss(projector, bio_batch, sd_batch, device)

            # Combined
            loss = loss_mse + args.lambda_cos * loss_cos + args.lambda_shift * loss_shift

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses["mse"] += loss_mse.item()
            train_losses["cos"] += loss_cos.item()
            train_losses["shift"] += loss_shift.item()
            train_losses["total"] += loss.item()
            n_batches += 1

        scheduler.step()

        # Average
        for k in train_losses:
            train_losses[k] /= n_batches

        # --- Validate ---
        projector.eval()
        val_losses = {"mse": 0, "cos": 0, "total": 0}
        n_val = 0

        with torch.no_grad():
            for bio_batch, sd_batch in val_loader:
                bio_batch = bio_batch.to(device)
                sd_batch = sd_batch.to(device)

                projected = projector(bio_batch)
                loss_mse = F.mse_loss(projected, sd_batch)
                cos_sim = F.cosine_similarity(projected, sd_batch, dim=-1)
                loss_cos = 1 - cos_sim.mean()

                val_losses["mse"] += loss_mse.item()
                val_losses["cos"] += loss_cos.item()
                val_losses["total"] += (loss_mse + args.lambda_cos * loss_cos).item()
                n_val += 1

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        # Log
        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train MSE={train_losses['mse']:.6f} Cos={train_losses['cos']:.4f} "
                f"Shift={train_losses['shift']:.6f} | "
                f"Val MSE={val_losses['mse']:.6f} Cos={val_losses['cos']:.4f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(projector.state_dict(), args.output)

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved best model to: {args.output}")


def evaluate(args):
    """Evaluate projection quality and shift preservation."""
    device = args.device

    data = torch.load(args.data, map_location="cpu")
    bio_embs = data["bio_embeddings"].to(device)
    sd_embs = data["sd_embeddings"].to(device)
    texts = data["texts"]

    projector = EmbeddingProjector(768).to(device)
    projector.load_state_dict(torch.load(args.checkpoint, map_location=device))
    projector.eval()

    with torch.no_grad():
        projected = projector(bio_embs)

        # Overall metrics
        mse = F.mse_loss(projected, sd_embs).item()
        cos_sim = F.cosine_similarity(projected, sd_embs, dim=-1).mean().item()
        print(f"Overall MSE: {mse:.6f}")
        print(f"Overall Cosine Similarity: {cos_sim:.4f}")

        # Find breast anchors for shift evaluation
        tumor_idx = None
        healthy_idx = None
        for i, t in enumerate(texts):
            if t == "tumor breast":
                tumor_idx = i
            elif t == "healthy breast":
                healthy_idx = i

        if tumor_idx is not None and healthy_idx is not None:
            bio_shift = projected[healthy_idx] - projected[tumor_idx]  # (77, 768)
            sd_shift = sd_embs[healthy_idx] - sd_embs[tumor_idx]  # (77, 768)

            shift_cos = F.cosine_similarity(
                bio_shift.unsqueeze(0), sd_shift.unsqueeze(0), dim=-1
            ).mean().item()
            shift_mse = F.mse_loss(bio_shift, sd_shift).item()

            print(f"\nShift Vector Preservation (tumor→healthy):")
            print(f"  Cosine Similarity: {shift_cos:.4f}")
            print(f"  MSE: {shift_mse:.6f}")
        else:
            print("Warning: 'tumor breast' or 'healthy breast' not found in texts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    # Train
    train_p = sub.add_parser("train", help="Train projection layer")
    train_p.add_argument("--data", type=str, default="./zero_shot_translation/paired_embeddings_medpix.pt")
    train_p.add_argument("--output", type=str, default="./zero_shot_translation/projector_medpix.pt")
    train_p.add_argument("--epochs", type=int, default=300)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--lambda-cos", type=float, default=0.5)
    train_p.add_argument("--lambda-shift", type=float, default=2.0)
    train_p.add_argument("--log-every", type=int, default=10)
    train_p.add_argument("--device", type=str, default="cuda")

    # Eval
    eval_p = sub.add_parser("eval", help="Evaluate projection")
    eval_p.add_argument("--data", type=str, default="./zero_shot_translation/paired_embeddings_medpix.pt")
    eval_p.add_argument("--checkpoint", type=str, default="./zero_shot_translation/projector_medpix.pt")
    eval_p.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        parser.print_help()
