import cv2
import numpy as np
import easyocr
import time

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Pass the language you need

cap = cv2.VideoCapture(1)  # 0 for default webcam

def get_package_coords_and_text():
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV

            # Define range for brown color in HSV
            lower_brown = np.array([10, 100, 20])
            upper_brown = np.array([20, 255, 200])

            # Threshold the HSV image to get only brown colors
            mask = cv2.inRange(hsv, lower_brown, upper_brown)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert to grayscale
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours are detected, then continue to the next frame
            if len(contours) == 0:
                continue

            # Assuming the largest contour will be your package.
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the coordinates of the center of the package
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:  # If the contour area is zero
                continue  # Skip this contour and continue to the next frame

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Crop the package region to get the text.
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = frame[y:y + h, x:x + w]
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # Run OCR on the cropped image
            result = reader.readtext(cropped_gray)
            text = ' '.join([res[1] for res in result])

            # Draw a circle in the center of the detected package on the image
            cv2.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the resulting frame
            # cv2.imshow('Package detection', frame)

            # If 'q' is pressed on the keyboard, exit this loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            return cX, cY, text

        # If no frame is captured, wait for a short period of time before retrying
        else:
            time.sleep(0.1)

    # Close the window and release webcam
    cap.release()
    cv2.destroyAllWindows()


while True:
    coords_and_text = get_package_coords_and_text()
    if coords_and_text:
        print(f"Coordinates: {coords_and_text[0]}, {coords_and_text[1]}")
        print(f"Text: {coords_and_text[2]}")
