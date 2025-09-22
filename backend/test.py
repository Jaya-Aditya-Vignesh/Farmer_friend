import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import sys

def load_model(pkl_path, device):
    model = torch.load(pkl_path, map_location=device)
    model.eval()
    model.to(device)
    return model

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    # Invert label map to map from integer index to label string
    idx_to_label = {v: k for k, v in label_map.items()}
    return idx_to_label

def preprocess_image(image_path, size=224):
    preprocessing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocessing(img).unsqueeze(0)  # Add batch dimension

def predict(model, input_tensor, idx_to_label):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    pred_idx = pred_idx.item()
    conf = conf.item()
    predicted_label = idx_to_label.get(pred_idx, "Unknown")

    # Optional: strip numeric prefix from label for clean output
    clean_label = "_".join(predicted_label.split('_')[1:])

    return clean_label, conf

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <model.pkl> <label_map.json> <image.jpg>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    label_map_path = sys.argv[2]
    image_path = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(pkl_path, device)
    idx_to_label = load_label_map(label_map_path)
    input_tensor = preprocess_image(image_path).to(device)

    pred_label, confidence = predict(model, input_tensor, idx_to_label)
    print(f"Predicted class: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
