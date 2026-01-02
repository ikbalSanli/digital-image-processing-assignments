import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gürültü Fonksiyonları
def add_gaussian_noise(img):
    sigma = 15
    gauss = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(float) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img):
    prob = 0.03
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def add_uniform_noise(img):
    low, high = -20, 20
    uni = np.random.uniform(low, high, img.shape)
    noisy = img.astype(float) + uni
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Filtre Fonksiyonları
def mean_filter(img):
    return cv2.blur(img, (5, 5))

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (9, 9), 3.0)

def median_filter(img):
    return cv2.medianBlur(img, 5)


def secure_choice(prompt, options):
    while True:
        c = input(prompt).strip()
        if c in options:
            return c
        print("Hatalı giriş, tekrar deneyin.")

def main():

    path = input("Görüntü yolunu girin: ").strip()
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("HATA: Görüntü yüklenemedi!")
        return

    print("\nGürültü tipi seçin:")
    print("1 - Gaussian gürültü")
    print("2 - Tuz-Biber gürültüsü")
    print("3 - Uniform gürültü")

    noise_choice = secure_choice("Seçim (1/2/3): ", ["1", "2", "3"])

    if noise_choice == "1":
        noisy = add_gaussian_noise(img)
        noise_name = "Gaussian (σ=15)"
    elif noise_choice == "2":
        noisy = add_salt_pepper_noise(img)
        noise_name = "Tuz-Biber (p=0.03)"
    else:
        noisy = add_uniform_noise(img)
        noise_name = "Uniform (-20, 20)"

    print("\nFiltre seçin:")
    print("1 - Mean filtre (5×5)")
    print("2 - Gaussian blur (9×9, σ=3)")
    print("3 - Median filtre (5×5)")

    filter_choice = secure_choice("Seçim (1/2/3): ", ["1", "2", "3"])

    if filter_choice == "1":
        filtered = mean_filter(noisy)
        filter_name = "Mean filtrasyon (5×5)"
    elif filter_choice == "2":
        filtered = gaussian_filter(noisy)
        filter_name = "Gaussian blur (9×9, σ=3)"
    else:
        filtered = median_filter(noisy)
        filter_name = "Median (5×5)"

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"); plt.title("Orijinal"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(noisy, cmap="gray"); plt.title("Gürültülü\n" + noise_name); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(filtered, cmap="gray"); plt.title("Filtrelenmiş\n" + filter_name); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
