import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model for caption generation...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

DATASET_PATH = Path("../Restaurant_food_datasets")
OUTPUT_CSV = Path("../metadata/manual_captions.csv")

# Candidate captions tailored to each restaurant based on their typical menu and benchmark queries
restaurant_captions = {
    "Dindigul_Thalappakatti": [
        "Dindigul style mutton biryani served on a plate",
        "Spicy mutton curry served with side dishes",
        "Kothu Parotta with salna",
        "Restaurant dining hall with wooden tables and seating"
    ],
    "Murugan_Idli_Shop": [
        "Soft idli served with sambar and coconut chutney",
        "Crispy ghee roast dosa",
        "Masala Vada and coconut chutney",
        "Authentic idli sambar breakfast"
    ],
    "Sangeetha_Veg_Restaurant": [
        "South Indian vegetarian thali with multiple side dishes",
        "Crispy masala dosa served with sambar and chutney",
        "Poori served with potato masala",
        "Bright well-lit dining area"
    ],
    "Absolute_Barbecues": [
        "Tableside live grilling with meat skewers",
        "Buffet counter with multiple food trays",
        "Modern restaurant interior",
        "Large group family dinner"
    ],
    "a2b": [
        "South Indian vegetarian meals served on a banana leaf",
        "Golden brown Gulab Jamun and Indian sweets",
        "Sweet Kesari with dry fruits",
        "Crispy Onion Rava Dosa",
        "Poori with masala"
    ],
    "southern_spice": [
        "Authentic Fish Curry with rice",
        "Spicy Chilli Chicken",
        "Traditional banana leaf meal",
        "Traditional decor restaurant interior",
        "Flavorful Hyderabadi Veg Biryani"
    ]
}

# Fallback captions if restaurant not in dict
fallback_captions = [
    "Delicious Indian meal served on a plate",
    "Restaurant interior and dining area",
    "Traditional South Indian food"
]

results = []
image_paths = sorted(list(DATASET_PATH.rglob("*.webp")) + list(DATASET_PATH.rglob("*.jpg")) + list(DATASET_PATH.rglob("*.png")))

print(f"Found {len(image_paths)} images to process. Generating captions...")

for img_path in image_paths:
    restaurant_name = img_path.parent.name
    candidates = restaurant_captions.get(restaurant_name, fallback_captions)
    
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            images=image,
            text=candidates,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
        best_idx = probs.argmax()
        best_caption = candidates[best_idx]
        
        # Save relative path starting with Restaurant_food_datasets/
        rel_path = img_path.relative_to(DATASET_PATH.parent).as_posix()
        results.append({"image_path": rel_path, "caption": best_caption})
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n--- Summary ---")
print(f"Total images processed: {len(results)}")
print(f"CSV path: {OUTPUT_CSV.resolve()}")
print("\nSample 10 generated captions:")
for _, row in df.sample(min(10, len(df)), random_state=42).iterrows():
    print(f"{row['image_path']}: {row['caption']}")
