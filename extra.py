import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fly = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED)


fw, fh = fly.shape[1] // 2, fly.shape[0] // 2

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.rectangle(frame, (cx - 80, cy - 80), (cx + 80, cy + 80), (255, 255, 255), 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 2000:
            (mx, my), r = cv2.minEnclosingCircle(c)
            mx, my, r = int(mx), int(my), int(r)

            if cx - 80 <= mx <= cx + 80 and cy - 80 <= my <= cy + 80:
                cv2.circle(frame, (mx, my), r, (0, 255, 0), 2)

                x1, y1 = mx - fw, my - fh
                if 0 <= x1 and 0 <= y1 and x1 + fw * 2 < w and y1 + fh * 2 < h:
                    alpha = fly[:, :, 3] / 255.0
                    for i in range(3):
                        frame[y1:y1 + fh * 2, x1:x1 + fw * 2, i] = \
                            (1 - alpha) * frame[y1:y1 + fh * 2, x1:x1 + fw * 2, i] + \
                            alpha * fly[:, :, i]

    cv2.imshow('Доп: муха', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()