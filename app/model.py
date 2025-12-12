import os
import wandb
from loadotenv import load_env 

# local folder and filename to save the model
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True)

load_env()

# TODO: wrap all the lines below in a function
def download_model_artifact():
    '''
    Function for fetching best model from Weights & Biases
    1. Login to wandb with API key from env variable
    2. Download the model artifact specified in the env variables
    3. Save the model artifact to the MODELS_DIR
    4. Print the download status
    5. Return nothing
    6. Raise an assertion error if WANDB_API_KEY is not found in env variables
    7. Use wandb.Api() to access the artifact
    '''
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY not found in environment variables"

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    api = wandb.Api()

    #artifact_path = "username/project_name/artifact_name:version"
    # compile model path from the env variables
    orga_name = os.getenv("WANDB_ORG")
    proj_name = os.getenv("WANDB_PROJECT")
    mod_name = os.getenv("WANDB_MODEL_NAME")
    mod_v = os.getenv("WANDB_MODEL_VERSION")
    artifact_path = f"{orga_name}/{proj_name}/{mod_name}:{mod_v}"
    artifact = api.artifact(artifact_path, type="model")
    print(f"Downloading artifact from {artifact_path}...")

    artifact.download(root=MODELS_DIR)

download_model_artifact()