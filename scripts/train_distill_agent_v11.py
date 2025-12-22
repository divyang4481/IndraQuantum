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
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
import torch.amp as amp
from datasets import load_dataset
from tqdm import tqdm
import copy
import sys

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v11 import IndraV11
from utils.logger_setup import setup_logger, MetricsLogger


class TopKDistillationLoss(nn.Module):
    def __init__(self, alpha_ce=0.5, alpha_kd=0.5, temperature=4.0, top_k=64):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.top_k = top_k
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets, attention_mask=None):
        # Shift so that tokens < n predict n
        shift_logits_student = student_logits[..., :-1, :].contiguous()
        shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
            flat_mask = shift_mask.view(-1)
        else:
            flat_mask = None

        # Flatten for CrossEntropy
        flat_logits_student = shift_logits_student.view(
            -1, shift_logits_student.size(-1)
        )
        flat_labels = shift_labels.view(-1)

        # --- 1. CE Loss (Hard Targets) ---
        # Only compute on unmasked labels (labels != -100)
        # Note: If -100 is used for prompt tokens, CE will ignore them automatically.
        # But we still apply mask if provided for padding.

        # Valid tokens are where label != -100 AND mask == 1 (if mask exists)
        # However, transformers DataCollator/Tokenizer usually sets pad labels to -100 too.
        # So relying on flat_labels != -100 is often sufficient, but we will be robust.

        loss_ce = F.cross_entropy(flat_logits_student, flat_labels, ignore_index=-100)

        # --- 2. KD Loss (Top-K Soft Targets) ---

        # We only want to compute KD on VALID tokens (not padding, not prompt if masked with -100)
        # We can treat all -100 labels as tokens to skip for KD too.
        valid_indices = flat_labels != -100

        if valid_indices.sum() == 0:
            return 0.0 * loss_ce, {
                "ce": loss_ce.item(),
                "kd": 0.0,
                "total": loss_ce.item(),
            }

        student_valid = flat_logits_student[valid_indices]
        teacher_valid = shift_logits_teacher.view(-1, shift_logits_teacher.size(-1))[
            valid_indices
        ]

        # Optimize: Top-K on Teacher
        with torch.no_grad():
            teacher_top_v, teacher_top_i = teacher_valid.topk(self.top_k, dim=-1)
            teacher_probs = F.softmax(teacher_top_v / self.temperature, dim=-1)

        # Gather Student Logits at Top-K indices
        student_top_v = student_valid.gather(1, teacher_top_i)
        student_log_probs = F.log_softmax(student_top_v / self.temperature, dim=-1)

        # KL Div
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            self.temperature**2
        )

        total_loss = (self.alpha_ce * loss_ce) + (self.alpha_kd * loss_kd)

        return total_loss, {
            "ce": loss_ce.item(),
            "kd": loss_kd.item(),
            "total": total_loss.item(),
        }


def load_teacher_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    print(f"Loading Teacher: {model_name}...")
    try:
        # Try loading on GPU first
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for teacher to save VRAM
            device_map="cuda",
            trust_remote_code=True,
        )
        print("Loaded Teacher on GPU (FP16).")
    except Exception as e:
        print(f"Failed to load on GPU: {e}. Fallback to CPU...")
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Loaded Teacher on CPU.")

    teacher.eval()
    return teacher


def initialize_embeddings_from_teacher(student, teacher, d_model=512):
    print("Initializing Student Embeddings from Teacher (Projection)...")
    with torch.no_grad():
        # Teacher Embeddings: [Vocab, T_Dim]
        t_emb = teacher.get_input_embeddings().weight.to(
            torch.float32
        )  # Cast to FP32 for projection
        t_dim = t_emb.shape[1]

        # Projection Matrix: [T_Dim, d_model]
        # We can use PCA or random orthogonal projection. Random orthogonal is easier and effective.
        # Ideally we want to preserve semantics.
        # Let's use a random projection for now as "Option 1" hack.
        projection = torch.randn(
            t_dim, d_model, device=t_emb.device, dtype=torch.float32
        )
        # Orthogonalize
        projection, _ = torch.linalg.qr(projection)

        # Projected: [Vocab, d_model]
        new_emb = torch.matmul(t_emb, projection)

        # Assign to Student (Complex Embedding has Real and Imag parts)
        # Initialize Real part with Projected, Imag part with small random or zero.
        # Normalize to avoid huge variance
        new_emb = new_emb / (d_model**0.5)

        student.embedding.embed_real.weight.data.copy_(new_emb)
        print("Student Embeddings Initialized.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup Logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{config['run_name']}_{timestamp}"
    run_dir = (
        os.path.join("runs", run_id)
        if not args.resume_checkpoint
        else os.path.dirname(args.resume_checkpoint)
    )
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir)
    metric_logger = MetricsLogger(run_dir, resume=bool(args.resume_checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # 1. Load Tokenizer
    teacher_name = config["distillation"]["teacher_model"]
    logger.info(f"Loading Tokenizer from {teacher_name}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL FIX 1: Truncation Side Left
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    logger.info("Set Tokenizer Truncation Side to LEFT (Keep Answer)")

    # 2. Teacher Model
    teacher = load_teacher_model(teacher_name)
    vocab_size = teacher.config.vocab_size

    # 3. Student Model
    logger.info("Initializing Student (IndraV11)...")
    student = IndraV11(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
        tie_word_embeddings=config["model"].get("tie_word_embeddings", False),
        gradient_checkpointing=config["model"].get("gradient_checkpointing", True),
    ).to(device)

    # CRITICAL FIX 5: Initialize Embeddings
    if not args.resume_checkpoint:
        initialize_embeddings_from_teacher(student, teacher, config["model"]["d_model"])

    # 4. Data
    logger.info("Loading Data...")
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["subset_name"],
        split="train",
        cache_dir="data_cache",
    )

    # CRITICAL FIX 2 & 3: Tokenize with Assistant Masking
    def tokenize(examples):
        # Optimized for OpenHermes / Chat structure
        inputs = []
        labels = []

        # Pre-compute special tokens
        pad_id = tokenizer.pad_token_id
        ignore_id = -100
        max_len = config["model"]["max_seq_len"]

        if "conversations" in examples:
            for conv in examples["conversations"]:
                # Construct Prompt (System + User) and Answer (Assistant)
                # We assume the last turn is Assistant.
                # OpenHermes: human/gpt or system/human/gpt

                # Separate messages
                msgs = []
                for m in conv:
                    role = "user" if m["from"] == "human" else "assistant"
                    if m["from"] == "system":
                        role = "system"
                    msgs.append({"role": role, "content": m["value"]})

                if msgs[-1]["role"] != "assistant":
                    # Skip if last msg is not assistant (rare but possible)
                    inputs.append([])
                    labels.append([])
                    continue

                # Everything but last msg
                prompt_msgs = msgs[:-1]
                answer_msg = msgs[-1]

                # Apply Template
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                answer_text = (
                    answer_msg["content"] + tokenizer.eos_token
                )  # Add EOS manually

                # Tokenize separately
                # Note: We need careful handling of max_len.
                # If total > max_len, we want to left-truncate PROMPT, keep ANSWER.
                # But tokenizer(truncation=True) handles full sequence.
                # Manual approach:

                prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
                answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

                # Concatenate
                total_ids = prompt_ids + answer_ids

                # Create mask (Prompt -> -100, Answer -> ID)
                # We want to train on Answer.
                # Labels: [-100, -100, ..., answer_id, answer_id]
                seq_labels = [ignore_id] * len(prompt_ids) + answer_ids

                # Truncate if too long (Left Truncate)
                if len(total_ids) > max_len:
                    # Keep end
                    total_ids = total_ids[-max_len:]
                    seq_labels = seq_labels[-max_len:]

                # Pad if too short
                pad_len = max_len - len(total_ids)
                if pad_len > 0:
                    total_ids = total_ids + [pad_id] * pad_len
                    seq_labels = (
                        seq_labels + [ignore_id] * pad_len
                    )  # Pad in labels is -100

                inputs.append(total_ids)
                labels.append(seq_labels)

        # Handle simple text if needed (fallback), but OpenHermes is convs
        else:
            # Just fallback to standard
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": [
                [1 if t != pad_id else 0 for t in ids] for ids in inputs
            ],
        }

    # Filter out empty entries from tokenization
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=config["data"].get("num_proc", 4),
    )
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    tokenized.set_format("torch")

    dataloader = DataLoader(
        tokenized,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        # Default collator handles converting lists to tensors, but we already have keys
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        pin_memory=True,
        num_workers=0,
    )

    # 5. Optimizer
    use_8bit = config["training"].get("use_8bit_optimizer", False)
    if use_8bit:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
            logger.info("Using 8-bit AdamW Optimizer (VRAM Saved!)")
        except ImportError:
            logger.warning("bitsandbytes not installed. Fallback to AdamW.")
            optimizer = optim.AdamW(
                student.parameters(), lr=float(config["training"]["learning_rate"])
            )
    else:
        optimizer = optim.AdamW(
            student.parameters(), lr=float(config["training"]["learning_rate"])
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=config["training"]["max_steps"],
    )

    # CRITICAL FIX 4: Top-K KD + High Temp
    criterion = TopKDistillationLoss(
        alpha_ce=config["distillation"]["alpha_ce"],
        alpha_kd=config["distillation"]["alpha_kd"],
        temperature=config["distillation"]["temperature"],
        top_k=config["distillation"].get("top_k", 64),
    )

    # Resume Logic
    optimizer_step = 0
    start_step = 0
    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        student.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        optimizer_step = start_step

    # Scaler
    scaler = amp.GradScaler("cuda", enabled=False)

    # Training Loop
    student.train()
    accum_steps = config["training"]["accumulate_grad_batches"]
    batch_step = optimizer_step * accum_steps

    # Schedule params
    max_steps = config["training"]["max_steps"]

    pbar = tqdm(total=max_steps, initial=optimizer_step, unit="step")

    start_time = time.time()

    for epoch in range(1000):
        for batch in dataloader:
            if optimizer_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # Has -100 for prompts
            mask = batch["attention_mask"].to(device)

            # --- Dynamic Schedule (Optional per user suggestion) ---
            # "first 2k steps: more CE... then ramp KD"
            # Let's adjust alpha dynamically if needed.
            # current_alpha_kd = min(config["distillation"]["alpha_kd"], optimizer_step / 2000.0)
            # For simplicity, we stick to fixed or implement if requested. User said "Use a schedule".
            # Simple Ramp:
            if optimizer_step < 2000:
                # Ramp KD from 0 to 0.5
                kd_factor = optimizer_step / 2000.0
                criterion.alpha_kd = config["distillation"]["alpha_kd"] * kd_factor
                criterion.alpha_ce = 1.0 - (0.5 * kd_factor)  # Balance
            else:
                criterion.alpha_kd = config["distillation"]["alpha_kd"]
                criterion.alpha_ce = config["distillation"]["alpha_ce"]

            with amp.autocast("cuda", enabled=False):
                with torch.no_grad():
                    # Move to Teacher Device
                    teacher_device = teacher.device
                    teacher_out = teacher(
                        input_ids.to(teacher_device),
                        attention_mask=mask.to(teacher_device),
                    )
                    teacher_logits = teacher_out.logits.to(device)

                student_logits, _, _ = student(input_ids, mask=mask)

                loss, loss_dict = criterion(
                    student_logits, teacher_logits, labels, attention_mask=mask
                )
                loss = loss / accum_steps

            loss.backward()
            batch_step += 1

            if batch_step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
                pbar.update(1)

                if optimizer_step % config["training"]["log_every"] == 0:
                    throughput = (
                        optimizer_step * config["training"]["batch_size"] * accum_steps
                    ) / (time.time() - start_time + 1e-6)
                    
                    # Console Metrics (Nice Formatting)
                    metrics = {
                        "loss": f"{loss_dict['total']:.3f}", 
                        "ce": f"{loss_dict['ce']:.3f}",
                        "kd": f"{loss_dict['kd']:.3f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}", 
                        "akd": f"{criterion.alpha_kd:.2f}",
                    }
                    
                    # CSV Metrics (Full Precision/Headers)
                    csv_metrics = {
                        "step": optimizer_step,
                        "epoch": epoch,
                        "total_loss": f"{loss_dict['total']:.4f}",
                        "ce_loss": f"{loss_dict['ce']:.4f}",
                        "phase_loss": "", 
                        "kd_loss": f"{loss_dict['kd']:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "throughput": f"{throughput:.2f}",
                    }
                    metric_logger.log(csv_metrics)
                    
                    # Update TQDM Postfix
                    pbar.set_postfix(metrics)

                if optimizer_step % config["training"]["save_every"] == 0:
                    ckpt_path = os.path.join(
                        run_dir, f"checkpoint_step{optimizer_step}.pt"
                    )
                    torch.save(
                        {
                            "step": optimizer_step,
                            "model_state_dict": student.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config,
                        },
                        ckpt_path,
                    )
                    logger.info(f"Saved {ckpt_path}")

    logger.info("Training Done.")
    torch.save(student.state_dict(), os.path.join(run_dir, "final_ IndraV11.pt"))


if __name__ == "__main__":
    main()
