import cv2
import numpy as np

def make_points(image, line):
    try:
        slope, intercept = line
        y1 = int(image.shape[0])  
        y2 = int(y1 * 3 / 5)      
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]
    except Exception as e:
        print(f"Error in make_points: {e}")
        return None

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:  
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:  
                left_fit.append((slope, intercept))
            else:  
                right_fit.append((slope, intercept))

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    else:
        left_line = None

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    else:
        right_line = None

    return left_line, right_line

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 130)  # change the values here, if its not having the right range of edges.
    return canny

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                try:
                    x1, y1, x2, y2 = line
                    cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
                except Exception as e:
                    print(f"Error in display_lines: {e}")
    return line_image

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([
        [(0, height),
         (width, height),
         (width, int(height/2)),
         (0, int(height/2))]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

cap = cv2.VideoCapture("/Users/floriangrassl/Documents/autonomousRCcar/random_german_Autobahn.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    if averaged_lines:
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
