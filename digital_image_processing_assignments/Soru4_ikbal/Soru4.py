import cv2
import numpy as np

img = cv2.imread("Soru4_ikbal/soru4.jpg", cv2.IMREAD_GRAYSCALE)

# --- 2) Median Blur ---
smooth = cv2.medianBlur(img, 5)

# --- 3) Histogram Eşitleme ---
hist_eq = cv2.equalizeHist(smooth)

# --- 4) Kontrast Germe ---
p5, p95 = np.percentile(hist_eq, (5, 95))
contrast = np.clip((hist_eq - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)

def add_label(image, text):
    labeled = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(labeled, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1, cv2.LINE_AA)
    return labeled

orig_labeled       = add_label(img,      "Original")
smooth_labeled     = add_label(smooth,   "Median Blur")
hist_labeled       = add_label(hist_eq,  "Histogram Equalization")
contrast_labeled   = add_label(contrast, "Contrast Stretching")
final_labeled      = add_label(contrast, "Final Output")

row1 = np.hstack((orig_labeled, smooth_labeled, hist_labeled))
row2 = np.hstack((contrast_labeled, final_labeled))

max_width = max(row1.shape[1], row2.shape[1])
pad1 = max_width - row1.shape[1]
pad2 = max_width - row2.shape[1]

row1 = np.hstack((row1, np.zeros((row1.shape[0], pad1, 3), dtype=np.uint8)))
row2 = np.hstack((row2, np.zeros((row2.shape[0], pad2, 3), dtype=np.uint8)))

grid = np.vstack((row1, row2))

cv2.imwrite("Soru4_sonuc.png", grid)

cv2.namedWindow("Restorasyon", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Restorasyon", 1200, 600)
cv2.imshow("Restorasyon", grid)

cv2.waitKey(0)
cv2.destroyAllWindows()
