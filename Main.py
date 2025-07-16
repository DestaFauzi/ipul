import cv2
import numpy as np
import pytesseract
import re
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Konfigurasi Tesseract (sesuaikan path jika diperlukan)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux

class LicensePlateDetector:
    def __init__(self):
        self.min_area = 1000  # Minimum area untuk deteksi plat
        self.max_area = 50000  # Maximum area untuk deteksi plat
        self.min_width = 80   # Minimum width plat
        self.max_width = 400  # Maximum width plat
        self.min_height = 20  # Minimum height plat
        self.max_height = 100 # Maximum height plat
        
        # Aspect ratio plat nomor Indonesia (biasanya 3:1 sampai 5:1)
        self.min_aspect_ratio = 2.0
        self.max_aspect_ratio = 6.0
        
        # Setup results directory
        self.results_dir = "detection_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def preprocess_image(self, image):
        """
        Preprocessing image untuk deteksi plat nomor
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter untuk mengurangi noise sambil mempertahankan edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply histogram equalization untuk meningkatkan kontras
        equalized = cv2.equalizeHist(filtered)
        
        # Edge detection menggunakan Canny
        edges = cv2.Canny(equalized, 30, 200)
        
        # Morphological operations untuk menutup gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return gray, filtered, equalized, edges
    
    def find_license_plate_contours(self, edges):
        """
        Mencari kontur yang berpotensi sebagai plat nomor
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        license_plate_candidates = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Skip if area is too small or too large
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check dimensions
            if (w < self.min_width or w > self.max_width or 
                h < self.min_height or h > self.max_height):
                continue
            
            # Check aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate extent (contour area / bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area
            
            # License plates usually have extent > 0.5
            if extent < 0.5:
                continue
            
            # Calculate solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                # License plates usually have solidity > 0.8
                if solidity < 0.8:
                    continue
            
            license_plate_candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity
            })
        
        return license_plate_candidates
    
    def extract_license_plate_roi(self, image, bbox):
        """
        Extract region of interest (ROI) dari plat nomor
        """
        x, y, w, h = bbox
        
        # Add some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        return roi
    
    def preprocess_for_ocr(self, roi):
        """
        Preprocessing khusus untuk OCR
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Resize untuk OCR yang lebih baik
        height, width = gray.shape
        if height < 40:
            scale_factor = 40 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 40), interpolation=cv2.INTER_CUBIC)
        
        # Apply gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations untuk membersihkan noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Invert jika background gelap
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
        
        return thresh
    
    def perform_ocr(self, processed_roi):
        """
        Melakukan OCR pada ROI yang sudah diproses
        """
        # Konfigurasi OCR untuk plat nomor
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(processed_roi, config=custom_config)
            
            # Clean the text
            text = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Get confidence scores
            data = pytesseract.image_to_data(processed_roi, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return text, avg_confidence
        
        except Exception as e:
            print(f"OCR Error: {e}")
            return "", 0
    
    def validate_license_plate(self, text):
        """
        Validasi format plat nomor Indonesia
        """
        if not text or len(text) < 5:
            return False
        
        # Pattern untuk plat nomor Indonesia
        patterns = [
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',  # Format umum
            r'^[A-Z]\d{4}[A-Z]{2}$',           # Format standar (B1234CD)
            r'^[A-Z]{2}\d{1,4}[A-Z]{1,2}$',   # Format 2 huruf depan
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def detect_license_plates(self, image):
        """
        Main function untuk deteksi plat nomor
        """
        # Preprocessing
        gray, filtered, equalized, edges = self.preprocess_image(image)
        
        # Find license plate candidates
        candidates = self.find_license_plate_contours(edges)
        
        results = []
        
        for candidate in candidates:
            # Extract ROI
            roi = self.extract_license_plate_roi(image, candidate['bbox'])
            
            # Preprocess for OCR
            processed_roi = self.preprocess_for_ocr(roi)
            
            # Perform OCR
            text, confidence = self.perform_ocr(processed_roi)
            
            # Validate license plate
            is_valid = self.validate_license_plate(text)
            
            if text and len(text) >= 5 and confidence > 30:  # Minimum confidence threshold
                results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': candidate['bbox'],
                    'roi': roi,
                    'processed_roi': processed_roi,
                    'is_valid': is_valid,
                    'contour': candidate['contour'],
                    'area': candidate['area'],
                    'aspect_ratio': candidate['aspect_ratio']
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results, {
            'gray': gray,
            'filtered': filtered,
            'equalized': equalized,
            'edges': edges
        }
    
    def draw_results(self, image, results):
        """
        Menggambar hasil deteksi pada image
        """
        output_image = image.copy()
        
        for i, result in enumerate(results):
            x, y, w, h = result['bbox']
            
            # Draw bounding box
            color = (0, 255, 0) if result['is_valid'] else (0, 255, 255)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw contour
            cv2.drawContours(output_image, [result['contour']], -1, color, 2)
            
            # Draw text
            text = result['text']
            confidence = result['confidence']
            label = f"{text} ({confidence:.1f}%)"
            
            # Calculate text position
            text_y = y - 10 if y > 30 else y + h + 25
            
            # Draw text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_image, (x, text_y - text_size[1] - 5), 
                         (x + text_size[0], text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(output_image, label, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_image
    
    def save_results(self, image, results, filename_prefix="detection"):
        """
        Menyimpan hasil deteksi
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main result
        main_result = self.draw_results(image, results)
        main_filename = f"{self.results_dir}/{filename_prefix}_{timestamp}.jpg"
        cv2.imwrite(main_filename, main_result)
        
        # Save individual ROIs
        for i, result in enumerate(results):
            roi_filename = f"{self.results_dir}/{filename_prefix}_{timestamp}_roi_{i}_{result['text']}.jpg"
            cv2.imwrite(roi_filename, result['roi'])
            
            processed_filename = f"{self.results_dir}/{filename_prefix}_{timestamp}_processed_{i}_{result['text']}.jpg"
            cv2.imwrite(processed_filename, result['processed_roi'])
        
        print(f"Results saved to {self.results_dir}")
        return main_filename

def main():
    parser = argparse.ArgumentParser(description='License Plate Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--camera', action='store_true', help='Use camera input')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--show-process', action='store_true', help='Show processing steps')
    args = parser.parse_args()
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    if args.image:
        # Process single image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        print("Processing image...")
        results, process_images = detector.detect_license_plates(image)
        
        # Display results
        output_image = detector.draw_results(image, results)
        
        # Show processing steps if requested
        if args.show_process:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original')
            axes[0, 1].imshow(process_images['gray'], cmap='gray')
            axes[0, 1].set_title('Grayscale')
            axes[0, 2].imshow(process_images['filtered'], cmap='gray')
            axes[0, 2].set_title('Filtered')
            axes[1, 0].imshow(process_images['equalized'], cmap='gray')
            axes[1, 0].set_title('Equalized')
            axes[1, 1].imshow(process_images['edges'], cmap='gray')
            axes[1, 1].set_title('Edges')
            axes[1, 2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title('Result')
            
            for ax in axes.flat:
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Display results
        cv2.imshow('License Plate Detection', output_image)
        
        # Print results
        if results:
            print(f"\nDetected {len(results)} license plate(s):")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['text']} (Confidence: {result['confidence']:.1f}%, Valid: {result['is_valid']})")
        else:
            print("No license plates detected.")
        
        # Save results if requested
        if args.save:
            detector.save_results(image, results, "image")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.video or args.camera:
        # Process video or camera
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Processing video... Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                results, _ = detector.detect_license_plates(frame)
                output_frame = detector.draw_results(frame, results)
                
                # Display frame info
                info_text = f"Frame: {frame_count}, Plates: {len(results)}"
                cv2.putText(output_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show current detections
                if results:
                    for i, result in enumerate(results):
                        if result['is_valid']:
                            detection_text = f"Valid: {result['text']} ({result['confidence']:.1f}%)"
                            cv2.putText(output_frame, detection_text, (10, 60 + i * 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                output_frame = frame
            
            cv2.imshow('License Plate Detection - Video', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save:
                detector.save_results(frame, results, f"video_frame_{frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Please specify input source: --image, --video, or --camera")
        parser.print_help()

if __name__ == "__main__":
    main()