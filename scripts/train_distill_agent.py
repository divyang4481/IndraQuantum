import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
import torch.amp as amp  # Use new namespace
from datasets import load_dataset
from tqdm import tqdm
import copy
import sys

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v5 import IndraV5
from utils.logger_setup import setup_logger, MetricsLogger


# --- DISTILLATION LOSS ---
class DistillationLoss(nn.Module):
    def __init__(self, alpha_ce=0.5, alpha_kd=0.5, temperature=2.0):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets, attention_mask=None):
        # Shift so that tokens < n predict n
        # Student Logits: [B, L, V] -> Use [:-1] to predict next
        # Targets: [B, L] -> Use [1:] as ground truth next

        shift_logits_student = student_logits[..., :-1, :].contiguous()
        shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        if attention_mask is not None:
            # shift mask similarly: [..., 1:] corresponds to the "next token" being valid
            shift_mask = attention_mask[..., 1:].contiguous()
            flat_mask = shift_mask.view(-1)
        else:
            flat_mask = None

        # Flatten for CrossEntropy
        # [B * (L-1), V]
        flat_logits_student = shift_logits_student.view(
            -1, shift_logits_student.size(-1)
        )
        flat_labels = shift_labels.view(-1)

        # 1. CE Loss (Hard Targets)
        if flat_mask is not None:
            # Mask labels (set to -100 where mask is 0)
            # We assume mask is 1 for valid, 0 for pad
            loss_ce_raw = F.cross_entropy(
                flat_logits_student, flat_labels, reduction="none"
            )
            loss_ce = (loss_ce_raw * flat_mask).sum() / (flat_mask.sum() + 1e-9)
        else:
            loss_ce = F.cross_entropy(flat_logits_student, flat_labels)

        # 2. KD Loss (Soft Targets)
        # Flatten teacher similarly
        flat_logits_teacher = shift_logits_teacher.view(
            -1, shift_logits_teacher.size(-1)
        )

        with torch.no_grad():
            teacher_probs = F.softmax(flat_logits_teacher / self.temperature, dim=-1)

        student_log_probs = F.log_softmax(
            flat_logits_student / self.temperature, dim=-1
        )

        # KLDiv "batchmean" divides by batch size.
        # Recalculate with reduction='none'
        loss_kd_raw = F.kl_div(
            student_log_probs, teacher_probs, reduction="none", log_target=False
        ) * (self.temperature**2)
        # sum over vocab -> [B*(L-1)]
        loss_kd_per_token = loss_kd_raw.sum(dim=-1)

        if flat_mask is not None:
            loss_kd = (loss_kd_per_token * flat_mask).sum() / (flat_mask.sum() + 1e-9)
        else:
            loss_kd = loss_kd_per_token.mean()

        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kd * loss_kd)

        return total_loss, {
            "ce": loss_ce.item(),
            "kd": loss_kd.item(),
            "total": total_loss.item(),
        }


def load_teacher_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading Teacher: {model_name} (CPU Offload to save VRAM)...")

    # Force CPU Loading to save 100% VRAM for Student
    # CPU inference is slower but allows Full FP32 Student Training on 6GB GPU
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU prefers float32
        device_map="cpu",
        trust_remote_code=True,
    )
    print("Loaded Teacher on CPU.")
    teacher.eval()  # Always Eval
    return teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    if args.resume_checkpoint:
        run_dir = os.path.dirname(args.resume_checkpoint)
        logger = setup_logger(run_dir)
        metric_logger = MetricsLogger(run_dir, resume=True)
        logger.info(f"Resuming run from: {run_dir}")
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{config['run_name']}_{timestamp}"
        run_dir = os.path.join("runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        logger = setup_logger(run_dir)
        metric_logger = MetricsLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # 1. Load Tokenizer (Use Teacher's Tokenizer)
    teacher_name = config["distillation"]["teacher_model"]
    logger.info(f"Loading Tokenizer from {teacher_name}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Teacher Model
    logger.info("Loading Teacher...")
    teacher = load_teacher_model(teacher_name)
    torch.cuda.empty_cache()

    # Use Teacher's Vocab Size to match Logits shape exactly
    vocab_size = teacher.config.vocab_size
    logger.info(f"Teacher Config Vocab Size: {vocab_size}")

    # 3. Student Model
    logger.info("Initializing Student (Indra Agent V1)...")
    student = IndraV5(
        vocab_size=vocab_size,  # Must match Teacher Logits
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
        tie_word_embeddings=config["model"].get("tie_word_embeddings", False),
        gradient_checkpointing=config["model"].get(
            "gradient_checkpointing", True
        ),  # Default to True for safety
    ).to(device)

    logger.info(
        f"Student Params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M"
    )

    # 4. Data
    logger.info("Loading Data...")
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["subset_name"],
        split="train",
        cache_dir="data_cache",  # Reuse cache
    )

    def tokenize(examples):
        # Handle OpenHermes 'conversations'
        if "conversations" in examples:
            texts = []
            for conv in examples["conversations"]:
                # Convert "from" to "role" if needed (OpenHermes uses from/value)
                # But Qwen template expects messages list.
                # Standardize specific to OpenHermes -> Qwen mapping
                messages = []
                for msg in conv:
                    role = "user" if msg["from"] == "human" else "assistant"
                    if msg["from"] == "system":
                        role = "system"
                    messages.append({"role": role, "content": msg["value"]})

                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
        elif "text" in examples:
            texts = examples["text"]
        else:
            # Fallback
            texts = [""] * len(examples[list(examples.keys())[0]])

        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config["model"]["max_seq_len"],
        )

    # Remove columns dynamically
    remove_cols = dataset.column_names
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=remove_cols,
        num_proc=config["data"].get("num_proc", 1),
    )
    tokenized.set_format("torch")

    dataloader = DataLoader(
        tokenized,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        pin_memory=True,  # Create pinned memory for faster transfer
        num_workers=0,  # Windows often fails with >0 workers in scripts, 0 is safer but slower.
    )

    # 5. Optimizer (configurable 8-bit)
    use_8bit = config["training"].get("use_8bit_optimizer", False)

    if use_8bit:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
            logger.info("Using 8-bit AdamW optimizer (saves ~1.2GB VRAM)")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
            optimizer = optim.AdamW(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
    else:
        # Try Paged 32-bit Optimizer to handle VRAM spillover (Shared RAM is slow)
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW32bit(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
            logger.info(
                "Using Paged 32-bit AdamW optimizer (Offloads states to RAM, restores speed)"
            )
        except ImportError:
            logger.warning(
                "bitsandbytes not available, using Standard AdamW (May spill to Shared RAM)"
            )
            optimizer = optim.AdamW(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=config["training"]["max_steps"],
    )

    criterion = DistillationLoss(
        alpha_ce=config["distillation"]["alpha_ce"],
        alpha_kd=config["distillation"]["alpha_kd"],
        temperature=config["distillation"]["temperature"],
    )

    # 6. EMA (optional)
    use_ema = config["training"].get("use_ema", False)
    ema_model = None
    if use_ema:
        ema_decay = config["training"].get("ema_decay", 0.9999)
        ema_model = copy.deepcopy(student)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info(f"Using EMA with decay={ema_decay}")

    # 7. Resume Logic
    start_step = 0
    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint: {args.resume_checkpoint}")
        # Map location to CPU to save VRAM spike
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")

        if "model_state_dict" in checkpoint:
            # Full State
            new_lr = config["training"]["learning_rate"]
            student.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )  # strict=False to allow adding new params like logit_scale
            # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # if "scheduler_state_dict" in checkpoint:
            #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(
                "Optimizer & Scheduler states RESET to support 8-bit -> 32-bit transition."
            )
            logger.info("A fresh Warmup phase will occur now.")

            start_step = checkpoint.get("step", 0)
            logger.info(f"Resumed from step {start_step} (Full State - Model Only)")

            if use_ema and "ema_state_dict" in checkpoint and ema_model:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])
                logger.info("EMA state restored.")
        else:
            # Legacy
            student.load_state_dict(checkpoint)
            logger.warning("Resumed MODEL ONLY (Optimizer reset). Loss may spike.")
            try:
                base = os.path.basename(args.resume_checkpoint)
                step_str = base.replace("checkpoint_step", "").replace(".pt", "")
                if "_" in step_str:
                    step_str = step_str.split("_")[0]
                start_step = int(step_str)
                logger.info(f"Inferred start step {start_step} from filename")
            except:
                start_step = 0

    # CLEANUP CHECKPOINT to save Memory
    if "checkpoint" in locals():
        del checkpoint
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # FORCE CONFIG LR (To allow tuning on resume)
    new_lr = config["training"]["learning_rate"]
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    logger.info(f"Optimizer LR forced to {new_lr:.2e} from config.")

    # AMP Scaler (New API) - DISABLED for Complex Weights Stability
    scaler = amp.GradScaler("cuda", enabled=False)

    # 8. Training Loop
    student.train()
    optimizer_step = start_step
    accum_steps = config["training"]["accumulate_grad_batches"]
    batch_step = optimizer_step * accum_steps

    logger.info(f"Starting Distillation from Opt Step {optimizer_step}...")
    start_time = time.time()

    # Progress bar
    pbar = tqdm(
        total=config["training"]["max_steps"],
        initial=optimizer_step,
        desc="Training",
        unit="opt_step",
    )

    for epoch in range(100):
        for batch in dataloader:
            if optimizer_step >= config["training"]["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # Ensure mask is present
            if "attention_mask" in batch:
                mask = batch["attention_mask"].to(device)
            else:
                mask = None

            # Mixed Precision Context - DISABLED to prevent ComplexHalf precision loss
            with amp.autocast("cuda", enabled=False):
                # TEACHER PASS (No Grad)
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids.to("cpu"),
                        attention_mask=mask.to("cpu") if mask is not None else None,
                    )
                    teacher_logits = teacher_out.logits.to(
                        device
                    )  # Move logits back to GPU for loss

                # STUDENT PASS
                student_logits, _, _ = student(input_ids, mask=mask)

                # LOSS
                loss, loss_dict = criterion(
                    student_logits, teacher_logits, labels, attention_mask=mask
                )
                loss = loss / accum_steps

            # Backward with Scaler
            scaler.scale(loss).backward()

            batch_step += 1

            # Show micro-batch progress
            micro_step = (batch_step - 1) % accum_steps + 1
            pbar.set_postfix(
                {
                    "micro": f"{micro_step}/{accum_steps}",
                    "total_loss": f"{loss_dict['total']:.4f}",
                    "ce_loss": f"{loss_dict['ce']:.4f}",
                    "kd_loss": f"{loss_dict['kd']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            # OPTIMIZER STEP
            if batch_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()

                optimizer_step += 1
                pbar.update(1)

                # EMA Update
                if use_ema:
                    with torch.no_grad():
                        for ema_param, student_param in zip(
                            ema_model.parameters(), student.parameters()
                        ):
                            ema_param.mul_(ema_decay).add_(
                                student_param, alpha=1 - ema_decay
                            )

                # Logging (based on optimizer steps)
                if optimizer_step % config["training"]["log_every"] == 0:
                    throughput = (
                        (optimizer_step - start_step)
                        * config["training"]["batch_size"]
                        * accum_steps
                    ) / (time.time() - start_time + 1e-6)
                    metrics = {
                        "step": optimizer_step,
                        "total_loss": f"{loss_dict['total']:.4f}",
                        "ce_loss": f"{loss_dict['ce']:.4f}",
                        "phase_loss": "",  # Not used in distillation
                        "kd_loss": f"{loss_dict['kd']:.4f}",  # KD loss
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "throughput": f"{throughput:.2f}_samples/s",
                    }
                    metric_logger.log(metrics)
                    logger.info(
                        f"Step {optimizer_step} | Total: {loss_dict['total']:.4f} | CE: {loss_dict['ce']:.4f} | KD: {loss_dict['kd']:.4f}"
                    )

                    if optimizer_step % (config["training"]["log_every"] * 50) == 0:
                        torch.cuda.empty_cache()

                # Checkpoint (based on optimizer steps)
                if optimizer_step % config["training"]["save_every"] == 0:
                    ckpt_path = os.path.join(
                        run_dir, f"checkpoint_step{optimizer_step}.pt"
                    )

                    save_dict = {
                        "step": optimizer_step,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config,
                    }
                    # Also save EMA inside the main dict or separate?
                    # Separate for now to keep 'model_state_dict' structure simple if user loads manually
                    if use_ema and ema_model:
                        save_dict["ema_state_dict"] = ema_model.state_dict()

                    torch.save(save_dict, ckpt_path)
                    logger.info(f"Saved STATE checkpoint: {ckpt_path}")

    pbar.close()

    # Save final models
    torch.save(student.state_dict(), os.path.join(run_dir, "final_agent_v1.pt"))
    tokenizer.save_pretrained(run_dir)
    logger.info("Saved Tokenizer")
    if use_ema:
        torch.save(
            ema_model.state_dict(), os.path.join(run_dir, "final_agent_v1_ema.pt")
        )
        logger.info("Saved final student AND EMA models")
    else:
        logger.info("Saved final student model")
    logger.info("Done.")


if __name__ == "__main__":
    main()
