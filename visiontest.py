import cv2
import numpy as np
import pytesseract
# Set the path to tesseract.exe in your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_package_and_read_label(frame):
     # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply adaptive thresholding to detect brown packages against black or white background
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center_coordinates = None
    label_text = ""
    roi = np.array([])

    # if contours exist, choose the one with maximum area
    if contours:
        contour = max(contours, key=cv2.contourArea)

        # apply contour approximation
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)

        # find the rotated bounding rectangle
        rect = cv2.minAreaRect(approx_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # calculate center of the bounding rectangle
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])

        # adjust these values as needed to best fit your labels
        roi_width = 200
        roi_height = 200

        roi = frame[max(0, center_y-roi_height//2):min(frame.shape[0], center_y+roi_height//2), max(0, center_x-roi_width//2):min(frame.shape[1], center_x+roi_width//2)]

        # convert ROI to gray and apply thresholding to isolate the label
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_threshold = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)

        # find contours in the thresholded ROI
        roi_contours, _ = cv2.findContours(roi_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # check if contours exist
        if roi_contours:
            # find the largest contour, assuming it's the label
            label_contour = max(roi_contours, key=cv2.contourArea)

            # calculate moments for the label contour
            M_label = cv2.moments(label_contour)

            # check if m00 is not zero (to avoid division by zero)
            if M_label["m00"] != 0:
                # calculate x,y coordinate of label center
                cX_label = int(M_label["m10"] / M_label["m00"])
                cY_label = int(M_label["m01"] / M_label["m00"])
                center_coordinates = (center_x + cX_label - roi_width//2, center_y + cY_label - roi_height//2)

                # draw a circle at the center
                cv2.circle(frame, center_coordinates, 10, (0, 255, 0), -1)

                # use pytesseract to extract text
                label_text = pytesseract.image_to_string(roi)

    return frame, threshold, roi, center_coordinates, label_text



cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    frame, threshold, roi, coords, text = detect_package_and_read_label(frame)

    # display the image with center marked
    cv2.imshow('Package Detection', frame)
    cv2.imshow('Threshold', threshold)

    # display the ROI if it is not empty
    if roi.size > 0:
        cv2.imshow('ROI', roi)

    print(f'Coordinates: {coords}\nLabel Text: {text}')

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()