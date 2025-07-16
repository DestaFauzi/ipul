# USAGE
# python Main.py --image Sample/s1.jpg  @ untuk file gambar
# python Main.py --video Sample/sv1.mp4  @ untuk file video
# python Main.py @ untuk cam

import argparse
import os
import re
import cv2
import time

import Calibration as cal
import DetectChars
import DetectPlates
import Preprocess as pp
import imutils
from ultrasonic_sensor import UltrasonicSensor

# Module level variables for image ##########################################################################

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)
SCALAR_ORANGE = (0.0, 165.0, 255.0)
N_VERIFY = 3

# Daftar plat nomor pemilik rumah
HOMEOWNER_PLATES = ["R3944FG", "R5477DP"]

# Ultrasonic sensor settings
ULTRASONIC_THRESHOLD = 15  # cm
TRIG_PIN = 18  # GPIO pin for trigger
ECHO_PIN = 24  # GPIO pin for echo

def clean_license_plate(raw_text):
    """
    Membersihkan hasil OCR plat nomor dengan logika yang aman
    """
    if not raw_text:
        return ""
    
    cleaned = raw_text.replace(" ", "").replace(".", "").replace("-", "").upper()
    
    print(f"Debug - Raw input: '{raw_text}' -> Cleaned: '{cleaned}'")
    
    if cleaned and cleaned[0].isalpha():
        print(f"Debug - Already starts with letter '{cleaned[0]}', no modification needed")
        return cleaned
    
    if cleaned and cleaned[0].isdigit():
        print(f"Debug - Starts with digit '{cleaned[0]}', checking for missing letter")
        
        if len(cleaned) >= 6:
            if cleaned == "3944FG":
                result = "R" + cleaned
                print(f"Debug - Detected missing 'R' for known plate: {result}")
                return result
            
            elif cleaned == "5477DP":
                result = "R" + cleaned
                print(f"Debug - Detected missing 'R' for known plate: {result}")
                return result
            
            elif re.match(r'^\d{4}[A-Z]{2}$', cleaned):
                result = "R" + cleaned
                print(f"Debug - Applied general pattern, added 'R': {result}")
                return result
    
    print(f"Debug - No modification applied, returning: '{cleaned}'")
    return cleaned

def validate_license_plate(plate_text):
    """
    Validasi apakah teks adalah plat nomor yang valid
    """
    if not plate_text or len(plate_text) < 5:
        return False
    
    patterns = [
        r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',
        r'^[A-Z]\d{4}[A-Z]{2}$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, plate_text):
            return True
    
    return False

def check_access(license_plate):
    """
    Fungsi untuk mengecek akses berdasarkan plat nomor
    """
    cleaned_plate = clean_license_plate(license_plate)
    
    if not validate_license_plate(cleaned_plate):
        return False, f"FORMAT TIDAK VALID: {cleaned_plate}", SCALAR_RED
    
    if cleaned_plate in HOMEOWNER_PLATES:
        return True, f"AKSES DIBERIKAN - Pemilik Rumah: {cleaned_plate}", SCALAR_GREEN
    else:
        return False, f"PERINGATAN - Bukan Pemilik Rumah: {cleaned_plate}", SCALAR_RED

def display_sensor_status(img, sensor, position=(10, 100)):
    """
    Menampilkan status sensor ultrasonik pada gambar
    """
    distance = sensor.get_distance()
    is_detected = sensor.is_object_detected(ULTRASONIC_THRESHOLD)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Status sensor
    if distance == float('inf'):
        distance_text = "No Object"
        status_color = SCALAR_WHITE
    else:
        distance_text = f"{distance:.1f}cm"
        status_color = SCALAR_GREEN if is_detected else SCALAR_ORANGE
    
    # Background untuk status
    status_text = f"Ultrasonic: {distance_text}"
    text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
    
    padding = 5
    bg_start = (position[0] - padding, position[1] - text_size[1] - padding)
    bg_end = (position[0] + text_size[0] + padding, position[1] + padding)
    
    cv2.rectangle(img, bg_start, bg_end, (0, 0, 0), -1)
    cv2.rectangle(img, bg_start, bg_end, status_color, 1)
    cv2.putText(img, status_text, position, font, font_scale, status_color, thickness)
    
    # Status deteksi
    detection_text = "OBJECT DETECTED - SCANNING..." if is_detected else "WAITING FOR OBJECT..."
    detection_color = SCALAR_GREEN if is_detected else SCALAR_YELLOW
    
    detection_pos = (position[0], position[1] + 30)
    cv2.putText(img, detection_text, detection_pos, font, font_scale * 0.8, detection_color, 1)

def display_access_message(img, message, color, position=(50, 150)):
    """
    Menampilkan pesan akses pada gambar
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    padding = 10
    bg_start = (position[0] - padding, position[1] - text_size[1] - padding)
    bg_end = (position[0] + text_size[0] + padding, position[1] + padding)
    
    cv2.rectangle(img, bg_start, bg_end, (0, 0, 0), -1)
    cv2.rectangle(img, bg_start, bg_end, color, 2)
    cv2.putText(img, message, position, font, font_scale, color, thickness)

def display_detection_info(img, license_plate, confidence=None):
    """
    Menampilkan informasi deteksi pada gambar
    """
    info_y = 200
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(img, f"Raw: {license_plate}", (10, info_y), font, 0.6, SCALAR_WHITE, 1)
    
    cleaned = clean_license_plate(license_plate)
    cv2.putText(img, f"Cleaned: {cleaned}", (10, info_y + 25), font, 0.6, SCALAR_YELLOW, 1)
    
    is_valid = validate_license_plate(cleaned)
    status_color = SCALAR_GREEN if is_valid else SCALAR_RED
    status_text = "VALID" if is_valid else "INVALID"
    cv2.putText(img, f"Status: {status_text}", (10, info_y + 50), font, 0.6, status_color, 1)
    
    if confidence:
        cv2.putText(img, f"Confidence: {confidence:.2f}", (10, info_y + 75), font, 0.6, SCALAR_WHITE, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    ap.add_argument("-i", "--image", help="Path to the image")
    ap.add_argument("-c", "--calibration", help="image or video or camera")
    ap.add_argument("--no-ultrasonic", action="store_true", help="Disable ultrasonic sensor")
    args = vars(ap.parse_args())

    img_original_scene = None
    loop = None
    camera = None
    ultrasonic_sensor = None

    # Initialize ultrasonic sensor (only for camera/video mode)
    use_ultrasonic = not args.get("no_ultrasonic", False) and not args.get("image", False)
    
    if use_ultrasonic:
        try:
            ultrasonic_sensor = UltrasonicSensor(TRIG_PIN, ECHO_PIN)
            ultrasonic_sensor.start_monitoring()
            print(f"Ultrasonic sensor activated - Threshold: {ULTRASONIC_THRESHOLD}cm")
        except Exception as e:
            print(f"Failed to initialize ultrasonic sensor: {e}")
            print("Continuing without ultrasonic sensor...")
            use_ultrasonic = False

    if args.get("calibration", True):
        img_original_scene = cv2.imread(args["calibration"])
        if img_original_scene is None:
            print("Please check again the path of image or argument !")
        img_original_scene = imutils.resize(img_original_scene, width=720)
        cal.calibration(img_original_scene)
        if ultrasonic_sensor:
            ultrasonic_sensor.cleanup()
        return
    else:
        if args.get("video", True):
            camera = cv2.VideoCapture(args["video"])
            if camera is None:
                print("Please check again the path of video or argument !")
            loop = True

        elif args.get("image", True):
            img_original_scene = cv2.imread(args["image"])
            if img_original_scene is None:
                print("Please check again the path of image or argument !")
                loop = False
        else:
            camera = cv2.VideoCapture(0)
            loop = True

    assert DetectChars.loadKNNDataAndTrainKNN(), "KNN can't be loaded !"

    save_number = 0
    prev_license = ""
    licenses_verify = []
    detection_count = 0
    last_scan_time = 0
    scan_cooldown = 2.0  # 2 seconds cooldown between scans

    if not os.path.exists("hasil"):
        os.makedirs("hasil")

    try:
        while loop:
            (grabbed, frame) = camera.read()
            if args.get("video") and not grabbed:
                break

            img_original_scene = imutils.resize(frame, width=620)
            
            # Display sensor status
            if use_ultrasonic and ultrasonic_sensor:
                display_sensor_status(img_original_scene, ultrasonic_sensor)
                
                # Check if object is detected and cooldown has passed
                current_time = time.time()
                object_detected = ultrasonic_sensor.is_object_detected(ULTRASONIC_THRESHOLD)
                cooldown_passed = (current_time - last_scan_time) > scan_cooldown
                
                if not object_detected:
                    # No object detected, skip license plate recognition
                    cv2.putText(img_original_scene, "SYSTEM STANDBY - NO OBJECT DETECTED", 
                               (10, img_original_scene.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, SCALAR_YELLOW, 2)
                    
                    cv2.putText(img_original_scene, "Press 'ESC' to exit", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow("imgOriginalScene", img_original_scene)
                    
                    key = cv2.waitKey(5) & 0xFF
                    if key == 27:
                        break
                    continue
                
                elif not cooldown_passed:
                    # Object detected but still in cooldown
                    remaining_cooldown = scan_cooldown - (current_time - last_scan_time)
                    cv2.putText(img_original_scene, f"COOLDOWN: {remaining_cooldown:.1f}s", 
                               (10, img_original_scene.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, SCALAR_ORANGE, 2)
                    
                    cv2.imshow("imgOriginalScene", img_original_scene)
                    
                    key = cv2.waitKey(5) & 0xFF
                    if key == 27:
                        break
                    continue
                
                else:
                    # Object detected and cooldown passed, proceed with license plate recognition
                    print(f"Object detected at {ultrasonic_sensor.get_distance():.1f}cm - Starting license plate recognition")
                    last_scan_time = current_time

            # Proceed with license plate recognition
            _, img_thresh = pp.preprocess(img_original_scene)
            cv2.imshow("threshold", img_thresh)

            img_original_scene = imutils.transform(img_original_scene)
            img_original_scene, new_license = searching(img_original_scene, loop)

            detection_count += 1

            if new_license:
                display_detection_info(img_original_scene, new_license)

            if new_license == "":
                if detection_count % 30 == 0:
                    print("no characters were detected\n")
            else:
                cleaned_license = clean_license_plate(new_license)
                
                if len(licenses_verify) == N_VERIFY and len(set([clean_license_plate(x) for x in licenses_verify])) == 1:
                    if clean_license_plate(prev_license) == cleaned_license:
                        print(f"still = {cleaned_license}\n")
                    else:
                        print(f"A new license plate read from image = {new_license} -> {cleaned_license}\n")
                        
                        access_granted, access_message, message_color = check_access(new_license)
                        print(access_message)
                        
                        display_access_message(img_original_scene, access_message, message_color, (50, 300))
                        
                        cv2.imshow(cleaned_license, img_original_scene)
                        file_name = f"hasil/{cleaned_license}.png"
                        cv2.imwrite(file_name, img_original_scene)
                        prev_license = new_license
                        licenses_verify = []
                else:
                    if len(licenses_verify) == N_VERIFY:
                        licenses_verify = licenses_verify[1:]
                    licenses_verify.append(new_license)

            cv2.putText(img_original_scene, "Press 's' to save frame, 'ESC' to exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detection area rectangle
            cv2.rectangle(img_original_scene,
                          ((img_original_scene.shape[1] // 2 - 230), (img_original_scene.shape[0] // 2 - 80)),
                          ((img_original_scene.shape[1] // 2 + 230), (img_original_scene.shape[0] // 2 + 80)), 
                          SCALAR_GREEN, 3)
            
            cv2.putText(img_original_scene, f"Homeowner plates: {', '.join(HOMEOWNER_PLATES)}", 
                       (10, img_original_scene.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, SCALAR_WHITE, 1)
            
            cv2.imshow("imgOriginalScene", img_original_scene)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('s'):
                save_number = str(save_number)
                savefileimg = f"calib_knn/img_{save_number}.png"
                savefileThr = f"calib_knn/Thr_{save_number}.png"
                
                if not os.path.exists("calib_knn"):
                    os.makedirs("calib_knn")
                    
                cv2.imwrite(savefileimg, frame)
                cv2.imwrite(savefileThr, img_thresh)
                print("image saved!")
                save_number = int(save_number) + 1
                
            if key == 27:
                break

        # For image only (no ultrasonic needed)
        if not loop:
            img_original_scene = imutils.resize(img_original_scene, width=720)
            cv2.imshow("original", img_original_scene)
            imgGrayscale, img_thresh = pp.preprocess(img_original_scene)
            cv2.imshow("threshold", img_thresh)
            img_original_scene = imutils.transform(img_original_scene)
            img_original_scene, new_license = searching(img_original_scene, loop)
            
            if new_license:
                cleaned_license = clean_license_plate(new_license)
                print(f"license plate read from image = {new_license} -> {cleaned_license}\n")
                
                display_detection_info(img_original_scene, new_license)
                
                access_granted, access_message, message_color = check_access(new_license)
                print(access_message)
                
                display_access_message(img_original_scene, access_message, message_color, (50, 150))
                
            cv2.waitKey(0)

    except KeyboardInterrupt:
        print("Program interrupted by user")
    
    finally:
        # Cleanup
        if camera:
            camera.release()
        if ultrasonic_sensor:
            ultrasonic_sensor.cleanup()
        cv2.destroyAllWindows()
        print("Program terminated")

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    raw_text = licPlate.strChars
    cleaned_text = clean_license_plate(raw_text)
    
    cv2.putText(imgOriginalScene, f"Raw: {raw_text}", (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), 
                intFontFace, fltFontScale * 0.8, SCALAR_YELLOW, intFontThickness)
    
    cv2.putText(imgOriginalScene, f"Clean: {cleaned_text}", 
                (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY + int(textSizeHeight * 1.2)), 
                intFontFace, fltFontScale * 0.8, SCALAR_WHITE, intFontThickness)

def searching(imgOriginalScene, loop):
    licenses = ""
    if imgOriginalScene is None:
        print("error: image not read from file \n")
        if os.name == 'nt':
            os.system("pause")
        return imgOriginalScene, licenses

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if not loop:
        cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        if not loop:
            print("no license plates were detected\n")
    else:
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlates[0]

        if not loop:
            cv2.imshow("imgPlate", licPlate.imgPlate)
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:
            if not loop:
                print("no characters were detected\n")
                return imgOriginalScene, licenses
        else:
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
            licenses = licPlate.strChars

            if not loop:
                cleaned = clean_license_plate(licenses)
                print("license plate read from image = " + licenses + f" -> {cleaned}\n")

        if not loop:
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    return imgOriginalScene, licenses

if __name__ == "__main__":
    main()
