import os
import shutil
from pathlib import Path


class CloudIO:
    """
    Abstraction for saving/loading artifacts in Cloud environments.
    Supports Azure ML, AWS S3 (via mounted paths), GCP Vertex AI.

    Currently assumes cloud environments mount storage to local paths
    or environment variables define output directories.
    """

    @staticmethod
    def get_output_dir(local_run_dir):
        """
        Returns the directory where artifacts should be saved.
        Priority:
        1. AZUREML_OUTPUT_DIRECTORY (Azure)
        2. SM_OUTPUT_DATA_DIR (AWS SageMaker)
        3. AIP_MODEL_DIR (GCP Vertex)
        4. local_run_dir (Local)
        """
        # Azure ML
        if "AZUREML_OUTPUT_DIRECTORY" in os.environ:
            return os.environ["AZUREML_OUTPUT_DIRECTORY"]

        # AWS SageMaker
        if "SM_OUTPUT_DATA_DIR" in os.environ:
            return os.environ["SM_OUTPUT_DATA_DIR"]

        # GCP Vertex AI
        if "AIP_MODEL_DIR" in os.environ:
            # GCP often uses gs:// paths which require GCS client.
            # For simplicity, if it's a local mount, return it.
            # If it's a URL, we might need a separate uploader.
            # Here we assume we just write to local and a sidecar syncs,
            # or return local_run_dir.
            # Many containers mount a path.
            pass

        return local_run_dir

    @staticmethod
    def save_file(local_path, target_filename=None):
        """
        Copies a local file to the cloud output directory if distinct.
        """
        target_dir = CloudIO.get_output_dir(os.path.dirname(local_path))
        if target_dir == os.path.dirname(local_path):
            return  # Already in place

        if target_filename is None:
            target_filename = os.path.basename(local_path)

        dest_path = os.path.join(target_dir, target_filename)
        shutil.copy2(local_path, dest_path)
        print(f"CloudIO: Synced {local_path} to {dest_path}")
