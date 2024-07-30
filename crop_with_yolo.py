from ultralytics import YOLO
import torch 
import concurrent.futures
from load_img_pdf import load_image
from constant import CONFIDENCE_TRESHOLD, CROP_EXPANSION
def crop_img(img, box, crop_expansion):
    """
    Crops an image based on a bounding box with optional expansion.

    Args:
    img (PIL.Image): The input image to be cropped.
    box (ultralytics.engine.results.Boxes): Bounding box object containing coordinates.
    crop_expansion (float): Factor to expand the crop area around the bounding box.

    Returns:
    PIL.Image: Cropped image.
    """
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    img_width, img_height = img.size
    width, height = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - width * crop_expansion / 2), max(0, y1 - height * crop_expansion / 2)
    x2, y2 = min(img_width, x2 + width * crop_expansion / 2), min(img_height, y2 + height * crop_expansion / 2)
    return img.crop((x1, y1, x2, y2))

def process_page(model, page, confidence_threshold, crop_expansion):
    """
    Processes a single page image with a YOLO model to detect and crop objects.

    Args:
    model (ultralytics.YOLO): Loaded YOLO model.
    page (PIL.Image): Page image to process.
    confidence_threshold (float): Minimum confidence score for object detection.
    crop_expansion (float): Factor to expand the crop area around detected objects.

    Returns:
    PIL.Image or None: Cropped image of the best detected object, or None if no object meets the threshold.
    """
    results = model(page)

    results = results[0]  
    if len(results.boxes) == 0:
        return None
    best_box = max(results.boxes, key=lambda x: x.conf.item())
    #print(best_box.conf.item())
    if best_box.conf.item() < confidence_threshold:
        
        return None
    return crop_img(page, best_box, crop_expansion)

def crop_with_yolo(path, model_path, confidence_threshold=None, crop_expansion=None):
    """
    Processes multiple pages from a file, detecting and cropping objects using a YOLO model.

    Args:
    path (str): Path to the input file (PDF or image).
    model_path (str): Path to the YOLO model weights.
    confidence_threshold (float, optional): Minimum confidence score for object detection. 
                                            Defaults to CONFIDENCE_TRESHOLD if not specified.
    crop_expansion (float, optional): Factor to expand the crop area around detected objects. 
                                      Defaults to CROP_EXPANSION if not specified.

    Returns:
    tuple: A tuple containing:
           - list of PIL.Image: Cropped images of detected objects.
           - list of PIL.Image: Original input pages.
    """
    if confidence_threshold == None:
        confidence_threshold = CONFIDENCE_TRESHOLD
    if crop_expansion == None:
        crop_expansion = CROP_EXPANSION
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')

    pages = load_image(path) 
    cropped_imgs = []
    for page in pages:
        coppred_img = process_page(model, page, confidence_threshold, crop_expansion)
        if coppred_img != None:
            cropped_imgs.append(coppred_img)
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #   futures = [executor.submit(process_page, model, page, confidence_threshold, crop_expansion) for page in pages]
    #    cropped_imgs = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]

    return cropped_imgs, pages