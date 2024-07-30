from pdf2image import convert_from_path
import fitz
from PIL import Image
import os 
import io 
from constant import PDF_EXTENSIONS, IMG_EXTENSIONS
import unittest

# Convert PIL Image to byte array for Azure OCR processing
def convert_with_pdf2image(pdf_path) -> list[Image.Image]:
    """
    Convert a PDF file into a list of PIL Image objects.

    Args:
        pdf_path (str): The path to the PDF file to be converted to images.

    Returns:
        list[Image.Image]: A list of PIL Image objects, where each element represents a page from the PDF file.

    Raises:
        FileNotFoundError: If the specified PDF file doesn't exist.
        ValueError: If the specified file is not a PDF.

    Example:
        images = convert_with_pdf2image("sample.pdf")
        print(len(images))  # Prints the number of pages in the PDF
    """
    pdf_path = pdf_path.strip()
    if not os.path.isfile(pdf_path):
        print(f"PDF file '{pdf_path}' doesn't exist.")
        raise FileNotFoundError
    elif os.path.splitext(pdf_path)[1].lower() != '.pdf':
        raise ValueError("File must be a PDF.")
    else:
        images = convert_from_path(pdf_path, output_folder=None)
        return images

def convert_with_pymupdf(pdf_path, dpi=300):
    """
    Convert a PDF file into a list of PIL Image objects using PyMuPDF.
    Much more faster than convert_with_pdf2image, but bad quality, which impact the accuracy. 

    Args:
        pdf_path (str): The path to the PDF file to be converted to images.
        dpi (int, optional): The resolution for the output images. Defaults to 300.

    Returns:
        list[Image.Image]: A list of PIL Image objects, where each element represents a page from the PDF file.

    Raises:
        FileNotFoundError: If the specified PDF file doesn't exist.
        ValueError: If the specified file is not a PDF.

    Example:
        images = convert_with_pymupdf("sample.pdf", dpi=200)
        print(len(images))  # Prints the number of pages in the PDF
    """
    pdf_path = pdf_path.strip()
    if not os.path.isfile(pdf_path):
        print(f"PDF file '{pdf_path}' doesn't exist.")
        raise FileNotFoundError
    elif os.path.splitext(pdf_path)[1].lower() != '.pdf':
        raise ValueError("File must be a PDF.")
    else:
        images = []
        zoom = dpi / 72  # Default PDF resolution is 72 dpi
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes('png')
                
                # Convert bytes to Pillow Image
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
        return images

def load_image(path):
    """
    Load an image or PDF file and return it as a list of PIL Image objects.

    Args:
        path (str): The file path to the image or PDF.

    Returns:
        list[Image.Image]: A list containing one or more PIL Image objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is neither a supported image format nor a PDF.

    Note:
        Supported formats are defined by PDF_EXTENSIONS and IMG_EXTENSIONS.

    Example:
        images = load_image("document.pdf")
        print(f"Loaded {len(images)} pages/images")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in PDF_EXTENSIONS:
        #images_data = convert_with_pymupdf(path)
        images_data = convert_with_pdf2image(path) # 2 choices, convert_with_pdf2image is better but slower
    elif ext in IMG_EXTENSIONS:
        images_data = [Image.open(path)]
    else: 
        raise ValueError(f"File is neither an image nor a PDF. Compatible formats are: {PDF_EXTENSIONS} and {IMG_EXTENSIONS}")
    
    return images_data

def load_image_resized(image_data, max_size=20971520):
    """
    Resize an image to ensure its size in bytes does not exceed a specified maximum.

    Args:
        image_data (bytes): The original image data as bytes.
        max_size (int, optional): The maximum allowed size in bytes. Defaults to 20971520 (20MB).

    Returns:
        bytes: The resized image data as bytes.

    Note:
        This function maintains the aspect ratio of the original image.
        The image is saved as JPEG with 85% quality after resizing.

    Example:
        with open("large_image.jpg", "rb") as f:
            original_data = f.read()
        resized_data = load_image_resized(original_data, max_size=10485760)  # 10MB max
        print(f"Original size: {len(original_data)}, Resized: {len(resized_data)}")
    """
    image = Image.open(io.BytesIO(image_data))

    # Calculate the scaling factor
    factor = (max_size / len(image_data)) ** 0.5
    
    # Calculate new dimensions
    new_width = int(image.width * factor)
    new_height = int(image.height * factor)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Save the resized image to a bytes buffer
    buffer = io.BytesIO()
    resized_image.save(buffer, format="JPEG", optimize=True, quality=85)
    resized_data = buffer.getvalue()
    
    return resized_data

def is_image_too_large(image_data, max_size=20971520):
    #TODO normalement, ne pose pas de pb dans le cas standard d'utilisation 
    #size = len(image_data)
    #return size > max_size 
    return False 

class TestOCRFunctions(unittest.TestCase):


    def test_convert_with_pdf2image(self):
        # This test requires a sample PDF file
        sample_pdf = "sample.pdf"
        if os.path.exists(sample_pdf):
            images = convert_with_pdf2image(sample_pdf)
            self.assertIsInstance(images, list)
            self.assertTrue(all(isinstance(img, Image.Image) for img in images))
        else:
            self.skipTest("Sample PDF not found for testing")