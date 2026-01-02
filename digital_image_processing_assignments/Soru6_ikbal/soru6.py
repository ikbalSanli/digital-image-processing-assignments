import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# PSNR hesaplama
def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 10 * math.log10((PIXEL_MAX ** 2) / mse)


# Laplace ile keskinleştirme 
def laplace_sharpen(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    sharp = gray.astype(np.float32) - lap
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    return sharp, lap

# Laplacian of gaussian keskinleştirme
def log_sharpen(gray, ksize=5, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

    log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)

    sharp = gray.astype(np.float32) - log
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    return sharp, log

# HIGH-BOOST keskinleştirme
def high_boost_sharpen(gray, k=1.5, ksize=5, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

    mask = gray.astype(np.float32) - blur.astype(np.float32)

    hb = gray.astype(np.float32) + k * mask
    hb = np.clip(hb, 0, 255).astype(np.uint8)

    return hb, blur, mask


def main():
    img_path = "Soru6_ikbal\image.png"  
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    sharp_lap, lap = laplace_sharpen(gray)

    sharp_log, log_resp = log_sharpen(gray, ksize=5, sigma=1.0)

    sharp_hb, hb_blur, hb_mask = high_boost_sharpen(gray, k=1.5, ksize=5, sigma=1.0)

    psnr_lap = psnr(gray, sharp_lap)
    psnr_log = psnr(gray, sharp_log)
    psnr_hb  = psnr(gray, sharp_hb)

    print(f"PSNR (Orijinal / Laplace):          {psnr_lap:.2f} dB")
    print(f"PSNR (Orijinal / LoG):              {psnr_log:.2f} dB")
    print(f"PSNR (Orijinal / High-Boost k=1.5): {psnr_hb:.2f} dB")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Orijinal")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(sharp_lap, cmap="gray")
    plt.title(f"Laplace (PSNR = {psnr_lap:.2f} dB)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(sharp_log, cmap="gray")
    plt.title(f"LoG (PSNR = {psnr_log:.2f} dB)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(sharp_hb, cmap="gray")
    plt.title(f"High-Boost k=1.5 (PSNR = {psnr_hb:.2f} dB)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("Soru6_sonuc.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
