import os
import concurrent.futures
import torch
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_path
import tempfile
from time import time

 #### 1
def crop_img(img, box, crop_expansion):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    img_width, img_height = img.size
    width, height = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - width * crop_expansion / 2), max(0, y1 - height * crop_expansion / 2)
    x2, y2 = min(img_width, x2 + width * crop_expansion / 2), min(img_height, y2 + height * crop_expansion / 2)
    return img.crop((x1, y1, x2, y2))

def crop_pdf_with_yolo(pdf_path, model_path, confidence_threshold=0.1, crop_expansion=0.1):
    # Load YOLOv8
    #start = time()
    t = time()
    model = YOLO(model_path)
    temps["load_model"].append(time() - t )
    #print(f"Model loaded in {time() - start} seconds")
    cropped_imgs = []

    with tempfile.TemporaryDirectory() as temp_dir:
        t = time()
        pages = convert_from_path(pdf_path)
        temps["convert_from_path"].append(time() - t)

        t_ = time()
        for i, page in enumerate(pages):
            temp_image_path = os.path.join(temp_dir, f"page_{i}.jpg")
            page.save(temp_image_path, "JPEG")

            t = time()
            results = model(temp_image_path)[0]
            temps["inference"].append(time() - t)

            if len(results.boxes) == 0:
                print(f"Aucune détection trouvée pour la page {i+1}")
                continue

            best_box = max(results.boxes, key=lambda x: x.conf.item())

            if best_box.conf.item() < confidence_threshold:
                print(f"Aucune détection avec une confiance suffisante pour la page {i+1}")
                continue

            t = time()
            cropped_img = crop_img(page ,best_box, crop_expansion)
            temps["crop"].append(time() - t)
            cropped_imgs.append(cropped_img)
    temps["enumerate"].append(time() - t_ )
    return cropped_imgs, pages

####2
def process_page(model, page, confidence_threshold, crop_expansion):
    results = model(page)[0]  # Resize image to 640x640
    if len(results.boxes) == 0:
        return None
    best_box = max(results.boxes, key=lambda x: x.conf.item())
    if best_box.conf.item() < confidence_threshold:
        return None
    return crop_img(page, best_box, crop_expansion)

def crop_pdf_with_yolo_(pdf_path, model_path, confidence_threshold=0.1, crop_expansion=0.1):
    
    t = time()
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')
    temps["load_model"].append(time() - t)
    
    t = time()
    pages = convert_from_path(pdf_path, dpi=200)  # Lower DPI for faster conversion
    temps["convert_from_path"].append(time() - t)
    
    t_ = time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, model, page, confidence_threshold, crop_expansion) for page in pages]
        cropped_imgs = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]
    
    temps["enumerate"].append(time() - t_)
    
    return cropped_imgs, pages

# Usage
pdf_path = "/content/950-2024-00002426_concatenated.pdf"
model_path = "/content/best.pt"
temps = {}
temps["load_model"] = []
temps["convert_from_path"] = []
temps["enumerate"] = []
temps["inference"] = []
temps["crop"] = []
start = time()
cropped_imgs, pages = crop_pdf_with_yolo(pdf_path, model_path)
print(f"Time for 1 {time() - start :.2f}")
print(temps)


temps["load_model"] = []
temps["convert_from_path"] = []
temps["enumerate"] = []
temps["inference"] = []
temps["crop"] = []
start = time()
cropped_imgs, pages = crop_pdf_with_yolo_(pdf_path, model_path)
print(f"Time for 1 {time() - start :.2f}")
print(temps)