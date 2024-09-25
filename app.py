import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoImageProcessor, ViTForImageClassification
from datasets import load_dataset
import torch


dataset = load_dataset("images", trust_remote_code=True)
image = dataset["test"]["image"][3]

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

image_processor = AutoImageProcessor.from_pretrained("oschamp/vit-artworkclassifier")
model = AutoModelForImageClassification.from_pretrained("oschamp/vit-artworkclassifier")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])