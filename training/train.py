import os
import argparse
import yaml
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Add root to sys.path to ensure modules are found
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v5 import IndraV5
from indra.losses.holographic import HolographicLoss
from utils.logger_setup import setup_logger, MetricsLogger
from utils.cloud_io import CloudIO


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="training/config_v5.yaml", help="Path to config"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Setup Runs Directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{config['run_name']}_{timestamp}"

    # Local path
    local_run_dir = os.path.join("runs", run_id)

    # Determine actual Output Directory (Cloud vs Local)
    # If standard local run, this returns local_run_dir.
    # If cloud, returns the mounted output path.
    run_dir = CloudIO.get_output_dir(local_run_dir)

    # If run_dir is different (e.g. /opt/ml/output), we might still want to mirror or link.
    # But for simplicity, we just use the determined path as the primary artifact store.
    os.makedirs(run_dir, exist_ok=True)

    # If using cloud path, print it
    if run_dir != local_run_dir:
        print(f"Cloud Environment Detected. Saving artifacts to: {run_dir}")

    # 2. Loggers
    logger = setup_logger(run_dir)
    metric_logger = MetricsLogger(run_dir)
    logger.info(f"Starting V5 Training Run: {run_id}")

    # 3. Model & Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Tokenizer (Use GPT2 for generic 50k vocab or Config vocab)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 fix

    # Update vocab size in config if needed
    vocab_size = config["model"].get("vocab_size", tokenizer.vocab_size)
    if vocab_size != tokenizer.vocab_size:
        logger.warning(
            f"Config vocab ({vocab_size}) != Tokenizer vocab ({tokenizer.vocab_size}). using Tokenizer's."
        )
        vocab_size = tokenizer.vocab_size

    # Update config with actual vocab size used
    config["model"]["vocab_size"] = vocab_size

    # Save config to run dir (Save AFTER modifications)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    model = IndraV5(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
    ).to(device)

    logger.info(
        f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )

    # 4. Data
    logger.info("Loading Dataset...")

    # Check for cache_dir in config
    cache_dir = config["data"].get("cache_dir", None)
    if cache_dir:
        # Ensure it exists
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using local dataset cache: {cache_dir}")

    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["dataset_config"],
        split="train",
        cache_dir=cache_dir,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config["model"]["max_seq_len"],
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
    )

    # 5. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    criterion = HolographicLoss(
        rho_mag=config["loss"]["rho_mag"], rho_phase=config["loss"]["rho_phase"]
    )

    # 6. Training Loop
    model.train()
    step = 0
    total_loss_accum = 0.0

    start_time = time.time()

    logger.info("Beginning Training Loop...")

    for epoch in range(100):  # Infinite-ish loop, controlled by max_steps
        for batch in dataloader:
            if step >= config["training"]["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            # Labels for CLM are typically input_ids shifted.
            # Transformers DataCollator does this automatically in 'labels'
            labels = batch["labels"].to(device)

            # Forward
            logits, z_mag, z_phase = model(input_ids)

            # Compute Loss
            # Logits: [B, L, V], Labels: [B, L]
            # Note: HolographicLoss might need updates to accept (mag, phase) directly or we reconstruct?
            # loss_fn signature: (student_logits, student_z, ...)
            # We can reconstruct student_z roughly or update HolographicLoss API.
            # But wait, HolographicLoss mainly needs phase.
            # Ideally we pass (mag, phase) or the complex z_out.
            # In V5.forward we removed returning z_out for direct mag/phase return.
            # Let's reconstruct or pass z_out.
            # Actually, let's update HolographicLoss to take mag/phase directly OR
            # update V5 forward to *also* return z_out or we re-compose.
            # Re-composing: z = mag * exp(1j * phase)
            z_reconstructed = torch.polar(z_mag, z_phase)

            loss, loss_dict = criterion(logits, z_reconstructed, targets=labels)

            # Backward
            loss.backward()

            if (step + 1) % config["training"]["accumulate_grad_batches"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss_accum += loss_dict["total"]

            # Logging
            if step % config["training"]["log_every"] == 0:
                throughput = (step * config["training"]["batch_size"]) / (
                    time.time() - start_time + 1e-6
                )
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "total_loss": f"{loss_dict['total']:.4f}",
                    "ce_loss": f"{loss_dict['ce']:.4f}",
                    "phase_loss": f"{loss_dict['phase_clustering']:.4f}",
                    "mag_loss": f"{loss_dict['mag_reg']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "throughput": f"{throughput:.2f}_samples/s",
                }
                metric_logger.log(metrics)
                logger.info(
                    f"Step {step} | Loss: {loss_dict['total']:.4f} | R: {loss_dict['phase_clustering']:.4f} | M: {loss_dict['mag_reg']:.4f}"
                )

            # Save Checkpoint
            if step > 0 and step % config["training"]["save_every"] == 0:
                ckpt_path = os.path.join(run_dir, f"checkpoint_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            step += 1

        if step >= config["training"]["max_steps"]:
            break

    # Final Save
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))
    logger.info("Training Complete.")


if __name__ == "__main__":
    main()
