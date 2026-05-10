import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(x), int(y)

            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))

            if 0.6 < circularity < 1.2:
                color = (0, 255, 0)

                if cx <= 50 and cy <= 50:
                    color = (255, 0, 0)
                elif cx >= (w - 50) and cy >= (h - 50):
                    color = (0, 0, 255)

                cv2.circle(frame, (cx, cy), int(radius), color, 3)
                cv2.circle(frame, (cx, cy), 5, color, -1)

    cv2.rectangle(frame, (0, 0), (50, 50), (255, 255, 255), 1)
    cv2.rectangle(frame, (w - 50, h - 50), (w, h), (255, 255, 255), 1)

    cv2.imshow('Метка', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()