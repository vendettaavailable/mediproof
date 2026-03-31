import easyocr

# Load model once (IMPORTANT)
reader = easyocr.Reader(['en'])

def process_image(image_path):
    result = reader.readtext(image_path)

    # Extract only text
    text = " ".join([item[1] for item in result])

    return text