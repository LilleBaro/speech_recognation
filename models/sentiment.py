import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer

from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.pretrained.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        # Pour CamemBERT/RoBERTa, on utilise souvent le premier token (<s>) 
        # accessible via last_hidden_state[:, 0, :]
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

def predict(text, model, tokenizer, device, max_length=256):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    
    return prediction.item(), confidence.item()

if __name__ == "__main__":
    MODEL_NAME = "bert-base-uncased"
    CHECKPOINT_PATH = "mon_checkpoint.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertClassifier(MODEL_NAME, n_classes=3)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)

    sample_text = "Des aides du #gouvernement pour des #logements plus #écologiques."
    label, score = predict(sample_text, model, tokenizer, DEVICE)

    label_map = {0: "Negative", 1: "Neutral", 2:"Positive"}
    print(f"Result: {label_map.get(label, 'Unknown')} (Score: {score:.4f})")

