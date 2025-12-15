import logging
import os
import sys
import csv
import time
from pathlib import Path


def setup_logger(run_dir, name="IndraV5"):
    # 1. System Logger (Console + File)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_file = os.path.join(run_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class MetricsLogger:
    def __init__(self, run_dir, filename="metrics.csv", resume=False):
        self.file_path = os.path.join(run_dir, filename)
        self.headers = [
            "step",
            "epoch",
            "total_loss",
            "ce_loss",
            "phase_loss",
            "kd_loss",
            "lr",
            "throughput",
        ]

        # Initialize CSV only if not resuming or file doesn't exist
        if not resume or not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        # Else: file exists and resume=True, just append later

    def log(self, metrics_dict):
        """
        metrics_dict keys must match headers
        """
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [metrics_dict.get(h, "") for h in self.headers]
            writer.writerow(row)
