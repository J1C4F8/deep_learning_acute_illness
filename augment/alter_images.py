import os
import sys
import skimage
import cv2
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from augment.face_org import exportImage


def average_blur(img, size):
    kernel = np.ones((size, size), np.float32)/(size*size)
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def gaussian_blur(img, size):
    return cv2.GaussianBlur(img, (size, size), 0)


def median_filtering(img, size):
    return cv2.medianBlur(img, size)


def bilateral_filtering(img, size):
    return cv2.bilateralFilter(img, size, 75, 75)


def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def noise(img, mode):
    return skimage.util.random_noise(img, mode=mode)


def plotimage(img, title, r, c, i):
    plt.subplot(r, c, i)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


def plot_all(img):
    noise_types = ["gaussian", "localvar", "poisson", "speckle"]
    gamma_vals = [0.75, 1.25, 1.50]
    blur = [3, 5]
    plt.figure(figsize=(10, 10))
    r = 3
    c = 4
    i = 1
    plotimage(img, "original", r, c, i)
    for gamma in gamma_vals:
        i += 1
        gamma = gamma if gamma > 0 else 0.5
        adjusted = adjust_gamma(img, gamma=gamma)
        plotimage(adjusted, "gamma=" + str(gamma), r, c, i)
    for n in noise_types:
        i += 1
        noisy = noise(img, n)
        plotimage(noisy, n, r, c, i)
    for size in blur:
        i += 1
        blurred = gaussian_blur(img, size)
        plotimage(blurred, "gaussian" + str(size), r, c, i)
    for size in blur:
        i += 1
        blurred = bilateral_filtering(img, size)
        plotimage(blurred, "bilateral" + str(size), r, c, i)
    plt.tight_layout()
    plt.savefig("data/plots/altered_images_plot.png")
    # plt.show()


def alter_and_save(img, filename):
    noise_types = ["gaussian", "localvar", "poisson", "speckle"]
    # gamma_vals = [75, 100, 150]
    blur = [3, 5]
    filename = filename.split(".")[0]
    # for gamma in gamma_vals:
    # gamma = gamma/100
    # gamma = gamma if gamma > 0 else 0.5
    # adjusted = adjust_gamma(img, gamma=gamma)
    # exportImage(target, filename, "gamma-"+ str(gamma) + ".png", adjusted)
    adjusted = adjust_gamma(img, gamma=1.3)
    exportImage(target, filename, "gamma-" + str(1.3) + ".png", adjusted)
    for n in noise_types:
        noisy = noise(img, n)
        exportImage(target, filename, "noisy-" + str(n) + ".png", noisy)
    for size in blur:
        blurred = gaussian_blur(img, size)
        exportImage(target, filename, "gaussian-" +
                    str(size) + ".png", blurred)
    for size in blur:
        blurred = bilateral_filtering(img, size)
        exportImage(target, filename, "bilateral-" +
                    str(size) + ".png", blurred)


def flip_all(source_path):
    for f in os.listdir(source_path):
        if f.startswith('.'):
            continue

        full_path = os.path.join(source_path, f)
        # if "_left" in f:
        #     renamed = f.split("_")[0] + "_right_brightened_flipped.png"
        #     os.rename(full_path, os.path.join(source_path, renamed))
        if os.path.isfile(full_path) and "_right" not in f and "_left" not in f:
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(img, 1)
            noisy = gaussian_blur(flipped, 5)
            target = '/'.join(source_path.split("/")[2:])
            exportImage(target, f[:-4], "flipped" + ".png", noisy)


if __name__ == "__main__":

    source_path = "data/parsed/"
    training_sick_folder = "data/parsed/training_sick/"
    source_folders = ["rug_healthy", "rug_sick"]
    teraget_folders = ["training_healthy", "training_sick"]

    # flip and add noise to sick images
    for f in os.listdir(training_sick_folder):
        if f.startswith('.'):
            continue
        full_path = os.path.join(training_sick_folder, f)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if "eye" not in f:
                noisy = noise(img, 'gaussian')
                flipped = cv2.flip(noisy, 1)
                exportImage("" + "training_sick", f[:-4], "augmented.png", flipped)
    
    for s_folder,t_folder in zip(source_folders, teraget_folders):
        print("Augmenting ", s_folder, "images...")
        for f in os.listdir(source_path + s_folder):
            if f.startswith('.'):
                continue
            full_path = os.path.join(source_path + s_folder, f)
            if os.path.isfile(full_path):
                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                brightened = adjust_gamma(img, 1.3)
                exportImage("" + t_folder,
                            f[:-4], "brightened.png", brightened)
                if "eye" not in f and "healthy" not in full_path:
                    noisy = noise(brightened, 'gaussian')
                    flipped = cv2.flip(noisy, 1)
                    exportImage("" + t_folder,
                                f[:-4], "augmented.png", flipped)

    print("Finished!\n")
