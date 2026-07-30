import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer


class BertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-multilingual-cased", n_classes=3):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


def _load_tokenizer(repo_id):
    try:
        return AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
    except Exception:
        return AutoTokenizer.from_pretrained(repo_id)


def _load_checkpoint(repo_id, device):
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="mon_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    return checkpoint


def load_model_and_tokenizer(repo_id, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(repo_id)

    model = BertClassifier("distilbert-base-multilingual-cased", n_classes=3)
    state_dict = _load_checkpoint(repo_id, device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text, model=None, tokenizer=None, device=None, repo_id=None, max_length=256):
    if model is None or tokenizer is None:
        if repo_id is None:
            raise ValueError("repo_id is required when model/tokenizer are not provided")
        model, tokenizer, device = load_model_and_tokenizer(repo_id, device)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label_map = {0: "Négatif", 1: "Neutre", 2: "Positif"}
    return label_map[pred.item()], conf.item()


def run_hf_inference(repo_id, text):
    print(f"Chargement du modèle depuis : {repo_id}...")
    model, tokenizer, device = load_model_and_tokenizer(repo_id)
    return predict(text, model=model, tokenizer=tokenizer, device=device, max_length=256)


if __name__ == "__main__":
    repo_id = os.environ.get("HF_REPO_ID", "LilleBaro/DistilBert_sentiment_analysis")
    test_text = "Le service était excellent et le personnel très accueillant."
    result, score = run_hf_inference(repo_id, test_text)

    print(f"\nTexte : {test_text}")
    print(f"Sentiment : {result} (Confiance : {score:.2%})")
