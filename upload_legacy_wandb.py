import os
import wandb

# 1. Initialize your project context
run = wandb.init(
    entity="mw4yrjg25n-zumerhub",
    project="satellite-landcover-unet",
    name="legacy-baseline-v1",
    job_type="upload-old-model",
    notes="Uploading the model trained on 2026-06-19 before WandB integration."
)

# 2. Define the path to your historical model
model_path = r"C:\samcodebase\deeplearning\Trained_Models\model_v1\best_unet_satellite_model_v1.22026-06-19_12-56-31.keras"

if os.path.exists(model_path):
    # 3. Create a WandB Artifact for the model
    artifact = wandb.Artifact(
        name="unet-satellite-baseline", 
        type="model",
        metadata={
            "version": "1.2",
            "training_date": "2026-06-19",
            "description": "Baseline satellite landcover U-Net model checkpoints"
        }
    )
    
# """
# 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

# $env:WANDB_API_KEY="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"
    
# """

    # 4. Bind the file and upload it
    artifact.add_file(model_path)
    run.log_artifact(artifact)
    print("[+] Historical model successfully pushed to WandB Artifact Registry!")
else:
    print(f"[-] Error: Could not locate the file at {model_path}")

# 5. Close the tracking run cleanly
run.finish()