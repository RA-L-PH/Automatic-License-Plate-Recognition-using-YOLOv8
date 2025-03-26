from ultralytics import YOLO
import cv2
import os
import numpy as np
import util
from util import read_license_plate
import argparse

def process_image(image_path, output_dir, coco_model, license_plate_detector):
    """
    Process a single image to detect vehicles and license plates.
    
    Args:
        image_path (str): Path to the image file
        output_dir (str): Directory to save the processed image
        coco_model: YOLO model for vehicle detection
        license_plate_detector: YOLO model for license plate detection
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Create a results dictionary
    results = {}
    
    # Detect vehicles
    detections = coco_model(frame)[0]
    vehicles = [2, 3, 5, 7]  # Vehicle class IDs (car, motorcycle, bus, truck)
    vehicle_detections = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles and score > 0.3:
            vehicle_detections.append([x1, y1, x2, y2, score])
            # Draw vehicle bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # Find which vehicle the license plate belongs to
        vehicle_found = False
        for vehicle in vehicle_detections:
            xv1, yv1, xv2, yv2, _ = vehicle
            # Check if license plate is inside a vehicle bounding box
            if x1 > xv1 and y1 > yv1 and x2 < xv2 and y2 < yv2:
                vehicle_found = True
                # Draw license plate bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    # Add text to the image
                    cv2.putText(frame, 
                                f"{license_plate_text} ({license_plate_text_score:.2f})", 
                                (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, 
                                (0, 0, 255), 
                                2)
                break
    
    # Save the processed image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images for license plate detection')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, default='output', help='Directory to save processed images')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    
    # Process all images in the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(args.input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(args.input_dir, filename)
            process_image(image_path, args.output_dir, coco_model, license_plate_detector)

if __name__ == "__main__":
    main()