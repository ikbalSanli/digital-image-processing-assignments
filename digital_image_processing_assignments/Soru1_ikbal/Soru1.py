import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Soru1_ikbal\soru1_orj.png", cv2.IMREAD_GRAYSCALE)

# Adaptive Threshold
thresh = cv2.adaptiveThreshold(
    img,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,   
    cv2.THRESH_BINARY,            
    51,                           
    5                             
)

# Invert 
invert = cv2.bitwise_not(thresh)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("1) Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(thresh, cmap='gray')
plt.title("2) Adaptive Threshold")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(invert, cmap='gray')
plt.title("3) Inverted Image")
plt.axis('off')

plt.savefig("Soru1_sonuc.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
