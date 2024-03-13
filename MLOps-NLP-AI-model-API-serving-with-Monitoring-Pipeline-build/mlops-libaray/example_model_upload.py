from mlopslib import MLOpsGCSClient

import json
with open('mlops-project-416811-32f591675633.json', 'r', encoding='utf8') as f:
    GCP_KEY_FILE = json.load(f)


client = MLOpsGCSClient(GCP_KEY_FILE)

client.upload_model(
    bucket_name="mlops-ml-bucket",
    model_name="nlp-model",
    local_dir_path="../model",
)