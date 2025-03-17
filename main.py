from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
from torchvision import transforms
from models.rams import ram_swin_large
import io

app = FastAPI()

# Load the AI Model
model = ram_swin_large(pretrained=False)
checkpoint = torch.load("models/rams/ram_swin_large_14m.pth", map_location="cpu")
if "model" not in checkpoint:
    raise KeyError("The checkpoint does not contain a 'model' key.")
model.load_state_dict(checkpoint["model"])
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_output(output, threshold=0.3):
    """
    Convert model output to a list of detected item names.
    :param output: Raw model output
    :param threshold: Confidence score threshold (default: 30%)
    :return: List of detected item names
    """
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    labels_path = "models/rams/categories.txt"  # Path to object labels
    try:
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    if output.dim() != 2 or output.size(1) != len(labels):
        raise ValueError("Unexpected model output shape. Ensure the model and labels are aligned.")

    probs = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
    k = min(5, probs.size(1))  # Ensure k does not exceed the number of classes
    top_probs, top_labels = torch.topk(probs, k)  # Get top-k predictions

    detected_items = [
        labels[idx] for prob, idx in zip(top_probs.squeeze(), top_labels.squeeze())
        if prob.item() > threshold
    ]

    return detected_items

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    detected_items = process_output(output)

    return {"detected_items": detected_items}