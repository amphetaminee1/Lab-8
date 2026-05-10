import cv2
import numpy as np


def add_white_noise(image, intensity=40):
    noise = np.random.normal(0, intensity, image.shape).astype(np.int16)

    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

img = cv2.imread('variant-5.jpg')

if img is not None:
    result = add_white_noise(img, intensity=60)

    cv2.imwrite('white_noise_result.jpg', result)
    cv2.imshow('Белый шум', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Файл не найден!")