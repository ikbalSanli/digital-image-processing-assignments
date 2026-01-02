import cv2
import numpy as np

img = cv2.imread("Soru2_ikbal/image.png", cv2.IMREAD_GRAYSCALE)

# Histogram Eşitleme
hist_eq = cv2.equalizeHist(img)

# Kontrast Germe
min_val = img.min()
max_val = img.max()
contrast = ((img - min_val) * (255 / (max_val - min_val))).astype(np.uint8)

# Gamma Düzeltmesi Fonksiyonu
def gamma_correct(im, gamma):
    return (np.power(im / 255.0, gamma) * 255).astype(np.uint8)

gamma_04 = gamma_correct(img, 0.4)
gamma_10 = gamma_correct(img, 1.0)
gamma_16 = gamma_correct(img, 1.6)

def add_label(image, text):
    labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(labeled, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return labeled

img_labeled       = add_label(img,       "Original")
hist_labeled      = add_label(hist_eq,   "Histogram Equalization")
contrast_labeled  = add_label(contrast,  "Contrast Stretching")
gamma04_labeled   = add_label(gamma_04,  "Gamma = 0.4")
gamma10_labeled   = add_label(gamma_10,  "Gamma = 1.0")
gamma16_labeled   = add_label(gamma_16,  "Gamma = 1.6")

row1 = np.hstack((img_labeled, hist_labeled, contrast_labeled))
row2 = np.hstack((gamma04_labeled, gamma10_labeled, gamma16_labeled))
grid = np.vstack((row1, row2))

cv2.imwrite("Soru2_sonuc.png", grid)

cv2.imshow("Sonuc", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
