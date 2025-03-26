import string
import easyocr
import cv2
import numpy as np
import torch

# Proper CUDA initialization
def setup_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            # Try to set device - this works in newer PyTorch versions
            torch.cuda.set_device(0)
        except:
            # For older PyTorch versions or in case of error
            device = torch.device('cuda:0')
            torch.cuda.current_device()
            
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Optimize memory usage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    else:
        print("CUDA is not available. Using CPU.")
        return False

# Setup CUDA
use_gpu = setup_cuda()

# Initialize the OCR reader with GPU if available
reader = easyocr.Reader(['en'], gpu=use_gpu)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required Indian format.
    Indian format: 2 letters (state) + 2 digits (district) + 1-2 letters (series) + 4 digits (number)
    Examples: MH12AB1234, DL2CAB1234, KA01AB1234
    
    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Remove any spaces that might be in the text
    text = text.replace(" ", "")
    
    # Check if length is between 9-10 characters (standard Indian format)
    if len(text) < 8 or len(text) > 10:
        return False
    
    # Basic pattern check for Indian format
    # First 2 chars should be letters (state code)
    if not (text[0].isalpha() and text[1].isalpha()):
        return False
    
    # Next 2 chars should be digits (district code)
    if not (text[2].isdigit() and text[3].isdigit()):
        return False
    
    # Last part should have at least 4 digits (vehicle number)
    digit_count = sum(1 for c in text[-4:] if c.isdigit())
    if digit_count < 4:
        return False
    
    return True


def format_license(text):
    """
    Format the license plate text for Indian plates.
    
    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # Remove spaces
    text = text.replace(" ", "")
    
    # Convert to uppercase
    text = text.upper()
    
    # Replace commonly misrecognized characters
    # Convert 'O' to '0' in numeric positions, etc.
    result = ""
    for i, char in enumerate(text):
        # First two characters should be letters (state code)
        if i < 2 and char in dict_char_to_int:
            result += dict_int_to_char.get(dict_char_to_int[char], char)
        # Characters at positions 2-3 should be digits (district code)
        elif i in [2, 3] and char in dict_int_to_char:
            result += dict_char_to_int.get(char, char)
        # Last 4 characters should be digits (vehicle number)
        elif i >= len(text) - 4 and char in dict_int_to_char:
            result += dict_char_to_int.get(char, char)
        # Middle characters could be either (series code)
        else:
            result += char
    
    return result


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image with enhanced processing
    specifically optimized for Indian license plates.

    Args:
        license_plate_crop (numpy.ndarray): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    # Keep original for fallback
    original = license_plate_crop.copy()
    height, width = license_plate_crop.shape[:2]
    
    # Function to correct perspective
    def correct_perspective(img):
        # Ensure we're working with grayscale
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Find edges in the image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        # Find the largest contour which should be the license plate
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to find the corners
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we get 4 points, perform perspective transform
        if len(approx) == 4:
            # Get the 4 corners in the correct order
            pts = np.float32([approx[i][0] for i in range(4)])
            
            # Sort by sum of coordinates (top-left has smallest sum)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            
            # Sort remaining points by difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left
            
            # Compute new width and height
            w1 = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            w2 = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            new_width = max(int(w1), int(w2))
            
            h1 = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            h2 = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            new_height = max(int(h1), int(h2))
            
            # Enforce aspect ratio typical for Indian plates (roughly 1:2 to 1:4)
            if new_width / new_height < 2.0:
                new_width = int(new_height * 2.5)  # Typical Indian plate ratio
            
            dst = np.array([
                [0, 0],
                [new_width - 1, 0],
                [new_width - 1, new_height - 1],
                [0, new_height - 1]
            ], dtype="float32")
            
            # Calculate and apply the transformation
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (new_width, new_height))
            
            return warped
            
        return img
    
    # Try to correct perspective
    try:
        transformed = correct_perspective(license_plate_crop)
        # If transformation significantly changed the image, use it
        if transformed.shape[0] > 10 and transformed.shape[1] > 10:
            license_plate_crop = transformed
    except Exception:
        # Fallback to original if perspective transform fails
        pass
    
    # Multiple processing attempts with different techniques
    processed_images = []
    
    # 1. Resize for better OCR performance - larger for Indian plates with thin characters
    resized = cv2.resize(license_plate_crop, (width*3, height*3))
    processed_images.append(resized)
    
    # 2. Convert to grayscale if not already
    if len(license_plate_crop.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()
    processed_images.append(gray)
    
    # 3. Color filtering for Indian plates (typically white/yellow background with black text)
    if len(license_plate_crop.shape) > 2:
        # HSV color space conversion
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # White plate mask (private vehicles)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow plate mask (commercial vehicles)
        lower_yellow = np.array([15, 80, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered = cv2.bitwise_and(resized, resized, mask=combined_mask)
        
        # Convert filtered result to grayscale
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        processed_images.append(filtered_gray)
    
    # 4. Apply stronger contrast enhancement (CLAHE) for Indian plates
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)
    
    # 5. Apply sharpening for clearer characters
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    processed_images.append(sharpened)
    
    # 6. Binary thresholding (both regular and inverted)
    ret, binary = cv2.threshold(sharpened, 100, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(binary)
    
    ret, binary_normal = cv2.threshold(sharpened, 100, 255, cv2.THRESH_BINARY)
    processed_images.append(binary_normal)
    
    # 7. Otsu's thresholding
    ret, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # 8. Adaptive thresholding - multiple block sizes for Indian fonts
    for block_size in [11, 15, 19]:
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, block_size, 2)
        processed_images.append(adaptive)
    
    # 9. Better denoising + thresholding
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    ret, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(denoised_thresh)
    
    # 10. Morphological operations tailored for Indian characters
    kernel_close = np.ones((3, 3), np.uint8)
    morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    processed_images.append(morph_close)
    
    # Opening to remove small noise
    kernel_open = np.ones((2, 2), np.uint8)
    morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)
    processed_images.append(morph_open)
    
    # 11. Edge enhancement 
    edged = cv2.Canny(gray, 30, 150)
    processed_images.append(edged)
    
    # 12. Dilate edges to connect broken character parts
    dilated_edges = cv2.dilate(edged, np.ones((2, 2), np.uint8), iterations=1)
    processed_images.append(dilated_edges)
    
    # Try all processed images and select the best result
    best_text = None
    best_score = 0
    
    # Configure EasyOCR for better recognition with Indian plates
    reader_config = {
        'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'batch_size': 1,
        'paragraph': False,
        'min_size': 10,
        'text_threshold': 0.6,
        'link_threshold': 0.8,
        'mag_ratio': 1.5
    }
    
    for img in processed_images:
        # Add white border to help OCR detect characters at edges
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Try with different rotation angles for skewed plates
        for angle in [0, -3, 3, -5, 5]:
            if angle != 0:
                # Rotate image
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = img.copy()
            
            # Process with EasyOCR
            detections = reader.readtext(rotated)
            
            for detection in detections:
                bbox, text, score = detection
                text = text.upper().replace(' ', '')
                
                # Check if format matches Indian plate and score is higher
                if license_complies_format(text) and score > best_score:
                    best_text = text
                    best_score = score
    
    if best_text:
        return format_license(best_text), best_score
    
    # One more attempt with the original image if all else fails
    detections = reader.readtext(original)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    
    return None, None


# Simplified OCR function for faster processing

def read_license_plate_fast(license_plate_crop):
    """Faster license plate OCR function"""
    # Keep original image
    if len(license_plate_crop.shape) > 2:
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = license_plate_crop.copy()
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive threshold - only one instead of multiple
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Add border to help OCR
    processed = cv2.copyMakeBorder(adaptive, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    # Configure EasyOCR for better speed
    reader_config = {
        'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'batch_size': 1,
        'paragraph': False,
        'min_size': 10,
        'text_threshold': 0.6,
    }
    
    # Only one detection attempt
    detections = reader.readtext(processed, **reader_config)
    
    # Process results
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        
        if license_complies_format(text):
            return format_license(text), score
    
    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
