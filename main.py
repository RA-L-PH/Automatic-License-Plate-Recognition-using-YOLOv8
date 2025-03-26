from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import threading
from datetime import timedelta
import argparse
import gc

from sort.sort import *
from util import get_car, read_license_plate, read_license_plate_fast, write_csv

class LicensePlateProcessor:
    def __init__(self, video_path, output_csv='./test.csv', resolution=640, skip_frames=2):
        """
        Initialize the license plate processor with performance optimizations.
        
        Args:
            video_path (str): Path to the video file
            output_csv (str): Path to the output CSV file
            resolution (int): Processing resolution (smaller = faster)
            skip_frames (int): Process 1 frame every n frames (higher = faster)
        """
        self.video_path = video_path
        self.output_csv = output_csv
        self.resolution = resolution
        self.skip_frames = max(1, skip_frames)
        
        # Initialize device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Initialize use_half for faster processing
        self.use_half = False  # Explicitly disable half precision to fix the dtype error
        
        # Performance tweaks
        if torch.cuda.is_available():
            # Force CUDA initialization
            torch.cuda.init()
            # Optimize CUDA performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Memory optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Print GPU info for debugging
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            
        # Get video info
        self.video_info = self.get_video_info()
        if not self.video_info:
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Print video info
        self.print_video_info()
        
        # Initialize models
        print("Loading models...")
        self.load_models_optimized()
        
        # Initialize tracker
        self.mot_tracker = Sort()
        
        # Vehicle classes
        self.vehicles = [2, 3, 5, 7]  # Car, motorcycle, bus, truck
        
        # Results storage
        self.results = {}
        
        # Detected vehicles
        self.detected_vehicles = set()
        
    def get_video_info(self):
        """Extract information about the video file."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None
        
        info = {
            'filename': os.path.basename(self.video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': ''.join([chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        
        # Adjusted frame count for frame skipping
        info['processed_frame_count'] = info['frame_count'] // self.skip_frames
        
        # Estimated processing times
        if torch.cuda.is_available():
            processing_fps = 30 * (640 / self.resolution) * self.skip_frames
        else:
            processing_fps = 8 * (640 / self.resolution) * self.skip_frames
            
        info['estimated_processing_time'] = info['frame_count'] / processing_fps
        
        cap.release()
        return info
    
    def print_video_info(self):
        """Print video information and processing estimates."""
        print("=" * 50)
        print(f"Processing video: {self.video_info['filename']}")
        print(f"Resolution: {self.video_info['width']}x{self.video_info['height']}")
        print(f"Frame rate: {self.video_info['fps']:.2f} FPS")
        print(f"Total frames: {self.video_info['frame_count']}")
        print(f"Processing frames: {self.video_info['processed_frame_count']} (every {self.skip_frames} frame(s))")
        print(f"Duration: {timedelta(seconds=self.video_info['duration'])}")
        print(f"Format: {self.video_info['fourcc']}")
        
        # Hardware info
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Using CPU (processing will be slower)")
            
        print(f"Estimated processing time: {timedelta(seconds=int(self.video_info['estimated_processing_time']))}")
        print(f"Processing resolution: {self.resolution}x{self.resolution}")
        print("=" * 50)
    
    def load_models_optimized(self):
        print("Loading optimized models...")
        # Load models
        self.coco_model = YOLO('yolov8n.pt')
        self.license_plate_detector = YOLO('license_plate_detector.pt')
        
        # Move models to device but don't use half precision 
        for model in [self.coco_model, self.license_plate_detector]:
            try:
                model.to(self.device)
                # Comment out or remove the half precision conversion
                # if self.use_half:
                #     model.half()
            except Exception as e:
                print(f"Warning: Could not optimize model: {e}")
                
        print(f"Models loaded on device: {self.device}")
        
        # Warmup the models to initialize CUDA context
        if self.device != 'cpu':
            dummy_input = torch.zeros((1, 3, self.resolution, self.resolution), device=self.device)
            if self.use_half:
                dummy_input = dummy_input.half()
            
            try:
                print("Warming up models...")
                with torch.no_grad():
                    self.coco_model(dummy_input, verbose=False)
                    self.license_plate_detector(dummy_input, verbose=False)
            except Exception as e:
                print(f"Warmup failed (non-critical): {e}")
                
    def preprocess_frame(self, frame):
        """Preprocess frame for inference - resize to target resolution."""
        # Only resize if needed (smaller resolution = faster processing)
        if max(frame.shape[:2]) > self.resolution:
            scale = self.resolution / max(frame.shape[:2])
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            return cv2.resize(frame, new_size), scale
        return frame, 1.0
    
    def postprocess_results(self, frame_results, scale):
        """Scale detection coordinates back to original resolution."""
        if scale == 1.0:
            return frame_results
            
        # Scale coordinates back to original resolution
        for vehicle in frame_results['vehicles']:
            vehicle[:4] = [coord / scale for coord in vehicle[:4]]
            
        for plate in frame_results['plates']:
            plate[:4] = [coord / scale for coord in plate[:4]]
            
        return frame_results
    
    def process_frame(self, frame_nmr, frame):
        """Process a single frame to detect vehicles and license plates."""
        # Preprocess frame for faster inference
        small_frame, scale = self.preprocess_frame(frame)
        
        # Initialize results for this frame
        frame_results = {'vehicles': [], 'plates': []}
        
        try:
            # Detect vehicles with optimized parameters
            # Add conf parameter to filter low confidence detections early
            detections = self.coco_model(
                small_frame, 
                verbose=False, 
                device=self.device, 
                imgsz=self.resolution,
                conf=0.3,  # Set confidence threshold
                half=self.use_half  # Match model precision
            )[0]
            
            # Extract vehicle detections
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in self.vehicles:  # No need to check score again as we set conf above
                    frame_results['vehicles'].append([x1, y1, x2, y2, score])
            
            # Only run tracking and plate detection if vehicles were found
            if frame_results['vehicles']:
                # Track vehicles
                track_ids = self.mot_tracker.update(np.asarray(frame_results['vehicles']))
                
                # Detect license plates - only if vehicles were found
                license_plates = self.license_plate_detector(
                    small_frame, 
                    verbose=False, 
                    device=self.device, 
                    imgsz=self.resolution,
                    half=self.use_half  # Match model precision
                )[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    frame_results['plates'].append([x1, y1, x2, y2, score, class_id])
                    
                    # Assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                    
                    if car_id != -1:
                        # Scale coordinates back to original size
                        if scale != 1.0:
                            x1, y1, x2, y2 = [int(coord / scale) for coord in [x1, y1, x2, y2]]
                        
                        # Crop license plate from original frame (not the resized one)
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        
                        # Process license plate - convert to grayscale and threshold
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        
                        # Read license plate text
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                        
                        if license_plate_text is not None:
                            # Store results
                            if frame_nmr not in self.results:
                                self.results[frame_nmr] = {}
                                
                            # Scale car coordinates back if needed
                            if scale != 1.0:
                                xcar1, ycar1, xcar2, ycar2 = [int(coord / scale) for coord in [xcar1, ycar1, xcar2, ycar2]]
                                
                            self.results[frame_nmr][car_id] = {
                                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'license_plate': {
                                    'bbox': [x1, y1, x2, y2],
                                    'text': license_plate_text,
                                    'bbox_score': score,
                                    'text_score': license_plate_text_score
                                }
                            }
                            
                            self.detected_vehicles.add(car_id)
                
            # Dynamically adjust skip_frames based on scene complexity
            self.skip_frames = self.dynamic_frame_skip(frame_results)
                
        except Exception as e:
            print(f"\nError processing frame {frame_nmr}: {e}")
            
        return frame_nmr
    
    def process_frame_batch(self, frames, frame_nmrs):
        """Process a batch of frames at once for faster inference"""
        if not frames:
            return {}
        
        batch_results = {}
            
        # Preprocess frames
        preprocessed = []
        scales = []
        for frame in frames:
            small_frame, scale = self.preprocess_frame(frame)
            preprocessed.append(small_frame)
            scales.append(scale)
        
        try:
            # Run batch inference for vehicles (much faster than one-by-one)
            vehicle_detections = self.coco_model(
                preprocessed, 
                verbose=False,
                device=self.device,
                imgsz=self.resolution,
                conf=0.3,
                half=self.use_half,
                batch=len(preprocessed)  # Set batch size
            )
            
            # Batch inference for license plates
            plate_detections = self.license_plate_detector(
                preprocessed,
                verbose=False,
                device=self.device,
                imgsz=self.resolution,
                half=self.use_half,
                conf=0.45,  # Add higher confidence threshold to filter false positives
                batch=len(preprocessed)
            )
            
            # Process each frame's detections
            for i, (v_dets, p_dets, frame_nmr, frame, scale) in enumerate(
                zip(vehicle_detections, plate_detections, frame_nmrs, frames, scales)
            ):
                # Extract vehicle detections
                frame_vehicles = []
                for detection in v_dets.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in self.vehicles:
                        frame_vehicles.append([x1, y1, x2, y2, score])
                    
                # Only process if vehicles were found
                if frame_vehicles:
                    # Update vehicle tracking
                    track_ids = self.mot_tracker.update(np.asarray(frame_vehicles))
                    
                    # Process license plates
                    for license_plate in p_dets.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate
                        
                        # Filter false positives using size/aspect ratio
                        frame_height, frame_width = frame.shape[:2]
                        if not self.is_valid_license_plate(x1, y1, x2, y2, frame_width, frame_height):
                            continue
                        
                        # Get associated vehicle
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                        
                        if car_id != -1:
                            # Scale coordinates back for original frame
                            if scale != 1.0:
                                x1, y1, x2, y2 = [int(coord / scale) for coord in [x1, y1, x2, y2]]
                            
                            # Crop license plate
                            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                            
                            # Add to process_frame_batch method before OCR call
                            import os
                            if not os.path.exists("debug_crops"):
                                os.makedirs("debug_crops")
                            cv2.imwrite(f"debug_crops/plate_{frame_nmr}_{car_id}.jpg", license_plate_crop)
                            
                            # Process and OCR - use fast version
                            license_plate_text, license_plate_text_score = read_license_plate_fast(license_plate_crop)
                            
                            if license_plate_text is not None:
                                # Store results
                                if frame_nmr not in self.results:
                                    self.results[frame_nmr] = {}
                                    
                                # Scale car coordinates back
                                if scale != 1.0:
                                    xcar1, ycar1, xcar2, ycar2 = [int(coord / scale) for coord in [xcar1, ycar1, xcar2, ycar2]]
                                    
                                self.results[frame_nmr][car_id] = {
                                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                    'license_plate': {
                                        'bbox': [x1, y1, x2, y2],
                                        'text': license_plate_text,
                                        'bbox_score': score,
                                        'text_score': license_plate_text_score
                                    }
                                }
                                
                                self.detected_vehicles.add(car_id)
            
            # Add in process_frame_batch method after license plate detection
            # Around line 369 after getting plate_detections
            for i, p_dets in enumerate(plate_detections):
                # Debug license plate detections
                if len(p_dets.boxes.data) > 0:
                    print(f"Frame {frame_nmrs[i]}: Found {len(p_dets.boxes.data)} license plate candidates")
                    print(f"  First plate bbox: {p_dets.boxes.data[0][:4].tolist()}")
                    print(f"  Score: {p_dets.boxes.data[0][4].item():.3f}")
                            
        except Exception as e:
            print(f"Error in batch processing: {e}")
        
        return self.results

    def process_video(self):
        """Process the video with optimized performance."""
        # Start timing
        start_time = time.time()
        frames_processed = 0
        processing_fps = 0
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        
        # Process frames
        frame_nmr = -1
        processed_count = 0
        
        print("Starting video processing...")
        
        while True:
            frame_nmr += 1
            
            # Skip frames for faster processing
            if frame_nmr % self.skip_frames != 0:
                # Skip frame but still need to read it
                ret = cap.grab()
                if not ret:
                    break
                continue
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            self.process_frame(frame_nmr, frame)
            
            # Update progress
            frames_processed += 1
            processed_count += 1
            if processed_count % 10 == 0:
                # Calculate progress stats
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    processing_fps = frames_processed / elapsed_time
                    remaining_frames = self.video_info['processed_frame_count'] - frames_processed
                    eta = remaining_frames / processing_fps if processing_fps > 0 else 0
                    
                    progress = (frames_processed / self.video_info['processed_frame_count']) * 100
                    eta_str = str(timedelta(seconds=int(eta)))
                    
                    print(f"Progress: {progress:.1f}% ({frames_processed}/{self.video_info['processed_frame_count']} frames) | "
                          f"Speed: {processing_fps:.2f} FPS | ETA: {eta_str}", end='\r')
                          
                # Periodically clean GPU memory
                if torch.cuda.is_available() and processed_count % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            # Early termination
            if len(self.detected_vehicles) >= 5 and processed_count > 100:
                print("Early stopping: sufficient license plates detected")
                break
        
        # Clean up
        cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frames_processed / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 50)
        print(f"Processing complete: {frames_processed}/{self.video_info['processed_frame_count']} frames processed")
        print(f"Total processing time: {timedelta(seconds=int(total_time))}")
        print(f"Average processing speed: {avg_fps:.2f} FPS")
        print(f"Total frames with license plates: {len(self.results)}")
        print("=" * 50)
        
        # Write results
        print(f"Writing results to {self.output_csv}...")
        write_csv(self.results, self.output_csv)
        print(f"Results saved to {self.output_csv}")
        
        return self.results
    
    def process_video_parallel(self):
        """Process video frames in parallel for faster execution"""
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        
        # Pre-read frames to remove I/O bottleneck
        frames = []
        frame_numbers = []
        frame_nmr = -1
        
        print("Reading video frames...")
        while True:
            frame_nmr += 1
            if frame_nmr % self.skip_frames != 0:
                ret = cap.grab()
                if not ret:
                    break
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            frame_numbers.append(frame_nmr)
            
            # Limit memory usage by processing in batches
            if len(frames) >= 100:
                break
        
        cap.release()
        
        print(f"Processing {len(frames)} frames in parallel...")
        
        # Process frames in parallel
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            futures = [executor.submit(self.process_frame, num, frame) 
                      for num, frame in zip(frame_numbers, frames)]
            
            for future in futures:
                future.result()
        
        print(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")
        return self.results
    
    def process_video_parallel_batched(self):
        """Process video frames in parallel batches for maximum speed"""
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        
        # Set buffer size for faster reading
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 128)
        
        print("Processing video in parallel batches...")
        frame_nmr = -1
        frames_processed = 0
        
        # Process in batches
        batch_size = 16  # Optimal batch size for GPU processing
        
        while True:
            frames_batch = []
            frame_numbers_batch = []
            
            # Fill batch
            for _ in range(batch_size):
                # Skip frames based on skip_frames setting
                while True:
                    frame_nmr += 1
                    if frame_nmr % self.skip_frames == 0:
                        break
                    ret = cap.grab()
                    if not ret:
                        break
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frames_batch.append(frame)
                frame_numbers_batch.append(frame_nmr)
            
            # Process batch if not empty
            if frames_batch:
                self.process_frame_batch(frames_batch, frame_numbers_batch)
                frames_processed += len(frames_batch)
                
                # Print progress
                if frames_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frames_processed / elapsed if elapsed > 0 else 0
                    print(f"Processed {frames_processed} frames at {fps:.2f} FPS")
                    
                # Optimize memory usage
                if torch.cuda.is_available() and frames_processed % 200 == 0:
                    torch.cuda.empty_cache()
            else:
                break
        
        # Clean up
        cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Print results
        total_time = time.time() - start_time
        avg_fps = frames_processed / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 50)
        print(f"Processing complete: {frames_processed} frames processed")
        print(f"Total processing time: {timedelta(seconds=int(total_time))}")
        print(f"Average processing speed: {avg_fps:.2f} FPS")
        print(f"Total frames with license plates: {len(self.results)}")
        print("=" * 50)
        
        # Add at the end of process_video_parallel_batched method before writing CSV
        print(f"Total detections to write: {sum(len(frame_data) for frame_data in self.results.values())}")
        print("Sample detections:")
        for frame_nmr in list(self.results.keys())[:3]:  # Print first 3 frames
            print(f"  Frame {frame_nmr}: {len(self.results[frame_nmr])} vehicles")
            for car_id, data in self.results[frame_nmr].items():
                if 'license_plate' in data and 'text' in data['license_plate']:
                    print(f"    Car {car_id}: {data['license_plate']['text']} (score: {data['license_plate']['text_score']:.2f})")
        
        # Write results
        print(f"Writing results to {self.output_csv}...")
        write_csv(self.results, self.output_csv)
        print(f"Results saved to {self.output_csv}")
        
        return self.results
    
    @staticmethod
    def process_batch(processor, video_path, output_csv, resolution=640, skip_frames=2):
        """Static method for parallel batch processing."""
        processor = LicensePlateProcessor(video_path, output_csv, resolution, skip_frames)
        return processor.process_video()

    def dynamic_frame_skip(self, frame_results):
        """Dynamically adjust frame skip rate based on scene complexity"""
        vehicle_count = len(frame_results['vehicles'])
        
        # More vehicles = more complex scene = process more frames
        if vehicle_count > 5:
            return max(1, self.skip_frames - 1)  # Process more frames in complex scenes
        elif vehicle_count < 2:
            return min(10, self.skip_frames + 2)  # Skip more frames in simple scenes
        else:
            return self.skip_frames  # Keep current rate

    def is_valid_license_plate(self, x1, y1, x2, y2, frame_width, frame_height):
        """Filter license plates by size and aspect ratio to remove false positives."""
        # Calculate width and height
        w = x2 - x1
        h = y2 - y1
        
        # License plates should have width > height (typical aspect ratio 2:1 to 4:1)
        aspect_ratio = w / h if h > 0 else 0
        
        # Size checks (relative to frame size)
        min_area_ratio = 0.0005  # Minimum area as percentage of frame
        max_area_ratio = 0.05    # Maximum area as percentage of frame
        
        area = w * h
        frame_area = frame_width * frame_height
        area_ratio = area / frame_area
        
        # Filter based on aspect ratio and size
        return (1.5 < aspect_ratio < 6.0 and 
                min_area_ratio < area_ratio < max_area_ratio)

if __name__ == "__main__":
    # Configure processing settings directly in code
    video_path = "./sample2.mp4"  # Path to your video file
    output_csv = "./test.csv"  # Path for results
    resolution = 512              # Processing resolution (lower = faster)
    skip_frames = 4               # Process every Nth frame (higher = faster)
    use_batch_processing = True   # Use batch processing for faster GPU utilization
    
    # Print configuration
    print(f"Processing video: {video_path}")
    print(f"Resolution: {resolution} | Skip frames: {skip_frames}")
    print(f"Processing mode: {'Batch' if use_batch_processing else 'Sequential'}")
    
    # Create processor with optimized settings
    processor = LicensePlateProcessor(
        video_path=video_path,
        output_csv=output_csv,
        resolution=resolution,
        skip_frames=skip_frames
    )
    
    # Enable half precision by default for faster processing (if GPU available)
    # Uncomment the line below to disable half precision
    # processor.use_half = False
    
    # Process the video with the best method
    if use_batch_processing:
        processor.process_video_parallel_batched()
    else:
        processor.process_video()
        
    print(f"Results saved to: {output_csv}")