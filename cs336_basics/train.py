import numpy as np
import torch
import os
from typing import Tuple, IO, Any, BinaryIO

import argparse

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import (
    cross_entropy,
    lr_cosine_schedule,
    clip_grad_norm_,
)

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

def get_batch(
        dataset: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length - 1
    assert max_start >= 0, "Dataset too small for given context length"

    start_indices = np.random.randint(0, max_start + 1, size = batch_size)

    x = np.stack(
        [dataset[i : i + context_length] for i in start_indices]
    )
    y = np.stack(
        [dataset[i + 1 : i + context_length + 1] for i in start_indices]
    )

    x = torch.tensor(x, dtype = torch.long, device=device)
    y = torch.tensor(y, dtype = torch.long, device=device)

    return x, y

def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        iteration: int, 
        out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TransformerLM model.")

    # data
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--merges', type=str, required=True)
    parser.add_argument('--special_tokens', type=str, nargs='*', default=[])

    # model
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--rope_theta', type=float, default=10000.0)

    # optimization
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=1000)

    # logging / eval
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--val_batches', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='runs/cs336')

    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=500)

    # misc
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        args.special_tokens
    )
    vocab_size = len(tokenizer.vocab)

    # ------------------------------------------------------------
    # Dataset (memmap)
    # ------------------------------------------------------------
    train_data = np.memmap(args.train_data, dtype=np.int32, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.int32, mode='r')

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------
    # Resume from checkpoint (if any)
    # ------------------------------------------------------------
    start_step = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_step = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
        )
        print(f"Resumed from checkpoint @ step {start_step}")

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    writer = SummaryWriter(args.logdir)
    scaler = GradScaler(enabled=args.use_amp)

    total_steps = args.epochs * args.steps_per_epoch
    max_lr = args.max_lr or args.lr

    # ------------------------------------------------------------
    # Batch samplers (closure, fixes argument mismatch)
    # ------------------------------------------------------------
    def sample_train_batch():
        return get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )

    def sample_val_batch():
        return get_batch(
            val_data,
            args.batch_size,
            args.context_length,
            device,
        )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for step in range(start_step, total_steps):
        model.train()

        # ----- LR schedule -----
        lr = lr_cosine_schedule(
            step,
            total_steps,
            max_lr=max_lr,
            warmup_steps=args.warmup_steps,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ----- forward -----
        x, y = sample_train_batch()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.use_amp):
            logits = model(x)
            loss = cross_entropy(logits, y)

        # ----- backward -----
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # ----- train logging -----
        if (step + 1) % args.log_interval == 0:
            ppl = torch.exp(loss).item()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/ppl", ppl, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)

            print(
                f"[{step+1:6d}] "
                f"loss={loss.item():.4f} "
                f"ppl={ppl:.2f} "
                f"lr={lr:.2e}"
            )

        # ----- validation -----
        if (step + 1) % args.val_interval == 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(args.val_batches):
                    xv, yv = sample_val_batch()
                    lv = cross_entropy(model(xv), yv)
                    val_losses.append(lv.item())

            val_loss = float(np.mean(val_losses))
            val_ppl = float(np.exp(val_loss))

            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/ppl", val_ppl, step)

            print(
                f"        >> val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f}"
            )

        # ----- checkpoint -----
        if args.checkpoint_path and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                args.checkpoint_path,
            )
            print(f"Checkpoint saved @ step {step+1}")

    writer.close()


if __name__ == "__main__":
    main()
