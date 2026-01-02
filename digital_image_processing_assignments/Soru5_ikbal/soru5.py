import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Soru5_ikbal/image.png", cv2.IMREAD_GRAYSCALE)

#  Binary Threshold 
_, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Yuvarlakları ayırmak için Morfolojik Açma 
circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
circles = cv2.morphologyEx(bw, cv2.MORPH_OPEN, circle_kernel)

bw_no_circles = cv2.subtract(bw, circles)

# Çubuk kernel
vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
sticks_vert = cv2.morphologyEx(bw_no_circles, cv2.MORPH_OPEN, vert_kernel)

horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
sticks_horiz = cv2.morphologyEx(bw_no_circles, cv2.MORPH_OPEN, horiz_kernel)

diag_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
sticks_diag = cv2.morphologyEx(bw_no_circles, cv2.MORPH_OPEN, diag_kernel)

# Çubukların birleşimi 
sticks = sticks_vert | sticks_horiz | sticks_diag

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
circles_open = cv2.morphologyEx(circles, cv2.MORPH_OPEN, open_kernel)

plt.figure(figsize=(18,10))

plt.subplot(2,4,1)
plt.imshow(img, cmap="gray")
plt.title("Orijinal")
plt.axis("off")

plt.subplot(2,4,2)
plt.imshow(bw, cmap="gray")
plt.title("Binary (Threshold)")
plt.axis("off")

plt.subplot(2,4,3)
plt.imshow(circles, cmap="gray")
plt.title("Morfolojik Açma (Yuvarlaklar)")
plt.axis("off")

plt.subplot(2,4,4)
plt.imshow(bw_no_circles, cmap="gray")
plt.title("Yuvarlaksız Görüntü")
plt.axis("off")

plt.subplot(2,4,5)
plt.imshow(sticks_vert, cmap="gray")
plt.title("Dikey Çubuklar")
plt.axis("off")

plt.subplot(2,4,6)
plt.imshow(sticks_horiz, cmap="gray")
plt.title("Yatay Çubuklar")
plt.axis("off")

plt.subplot(2,4,7)
plt.imshow(sticks_diag, cmap="gray")
plt.title("Çapraz Çubuklar")
plt.axis("off")

plt.subplot(2,4,8)
plt.imshow(sticks, cmap="gray")
plt.title("Tüm Çubuklar (Birleşim)")
plt.axis("off")

plt.tight_layout()
plt.savefig("Soru5_tum_asamalar.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(bw, cmap="gray")
plt.title("Binary")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(circles_open, cmap="gray")
plt.title("Yuvarlaklar (Final)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(sticks, cmap="gray")
plt.title("Çubuklar (Final)")
plt.axis("off")

plt.savefig("Soru5_sonuc.png", dpi=300, bbox_inches='tight')
plt.show()
