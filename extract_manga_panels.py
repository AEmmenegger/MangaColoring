import os
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
from tqdm import tqdm
import fitz

def pdf_to_images(pdf_path, images_dir):
    """
    Converts each page of a PDF into an image and saves it.
    """
    os.makedirs(images_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi = 200)
    print(f"Converting {len(pages)} pages to images...")
    page_paths = []

    for idx, page in enumerate(pages):
        image_path = os.path.join(images_dir, f'page_{idx + 1:03d}.png')
        page.save(image_path, 'PNG')
        page_paths.append(image_path)

    return page_paths

def convert_pdf_to_image(pdf_path, images_dir):
    #open your file
    doc = fitz.open(pdf_path)
    #iterate through the pages of the document and create a RGB image of the page
    page_paths = []
    for idx, page in enumerate(doc):
        image_path = os.path.join(images_dir, f'page_{idx + 1:03d}.png')
        pix = page.get_pixmap()
        pix.save(image_path, 'PNG')
        page_paths.append(image_path)
    return page_paths

def detect_panels(page_image_path):
    """
    Detects rectangular panels in a manga page image.
    Returns the list of panel images and their coordinates.
    """
    img = cv2.imread(page_image_path)
    if img is None:
        print(f"Failed to read image: {page_image_path}")
        return [], []

    original_img = img.copy()

    # Add a border around the image
    border_size = 50  # Adjust this size as needed
    img_with_border = cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White border
    )

    # Convert to grayscale
    gray = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY)

    # Invert the image
    gray_inv = cv2.bitwise_not(gray)

    # Thresholding to get binary image
    _, thresh = cv2.threshold(gray_inv, 200, 255, cv2.THRESH_BINARY)

    # Dilate to connect text regions (optional)
    dilated = thresh

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_images = []
    panel_coords = []
    debug_counter = 0

    for cnt in contours:
        # Approximate the contour to get a rectangle
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # We assume panels are quadrilaterals
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Filter out small regions
            if w < 100 or h < 100:
                continue

            # Adjust coordinates to remove the border offset
            x_adj = max(x - border_size, 0)
            y_adj = max(y - border_size, 0)
            w_adj = min(w, original_img.shape[1] - x_adj)
            h_adj = min(h, original_img.shape[0] - y_adj)

            # Extract the panel from the original image
            panel_img = original_img[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
            panel_images.append(panel_img)
            panel_coords.append({
                'x': int(x_adj),
                'y': int(y_adj),
                'width': int(w_adj),
                'height': int(h_adj)
            })



    return panel_images, panel_coords
def process_manga(pdf_path, panels_dir, coords_json_path):
    """
    Processes the manga PDF to extract panels and save their coordinates.
    """
    images_dir = 'mangas/pages'
    os.makedirs(panels_dir, exist_ok=True)

    # Convert PDF pages to images
    #print("Converting PDF pages to images...")
    #page_images = convert_pdf_to_image(pdf_path, images_dir)
    #get page_images from the panels folder
    page_images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.png')]

    all_coords = {}

    for idx, page_image_path in enumerate(tqdm(page_images, desc='Processing pages')):
        page_number = idx + 1
        panel_images, panel_coords = detect_panels(page_image_path)

        # Save each panel image and record its coordinates
        page_panels_info = []
        for i, (panel_img, coords) in enumerate(zip(panel_images, panel_coords)):
            panel_filename = f'page_{page_number:03d}_panel_{i + 1:02d}.png'
            panel_path = os.path.join(panels_dir, panel_filename)
            cv2.imwrite(panel_path, panel_img)

            # Add the panel info
            panel_info = {
                'panel_filename': panel_filename,
                'coordinates': coords
            }
            page_panels_info.append(panel_info)

        # Add the page info
        all_coords[f'page_{page_number}'] = page_panels_info

    # Save the coordinates to a JSON file
    with open(coords_json_path, 'w') as json_file:
        json.dump(all_coords, json_file, indent=4)

    print(f"Processing complete. Panels saved in {panels_dir}. Coordinates saved in {coords_json_path}.")

if __name__ == '__main__':
    pdf_path = 'mangas/pdfs/'  # Replace with your PDF file path
    panels_dir = 'mangas/panels'
    coords_json_path = 'mangas/panel_coordinates.json'
    for pdf_file in os.listdir(pdf_path):
        if pdf_file.endswith('.pdf'):
            full_pdf_path = os.path.join(pdf_path, pdf_file)
            print(full_pdf_path)
            process_manga(full_pdf_path, panels_dir, coords_json_path)

    #process_manga(pdf_path, panels_dir, coords_json_path)
