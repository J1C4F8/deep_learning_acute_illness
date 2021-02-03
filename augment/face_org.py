'''
Script that calculates:
 * a circle enclosing each eye, along with its corresponding eyebrow,
 * a circle enclosing the lips,
 * the skin pixels, without eyebrows, eyes or mouth.
 
The right-eye regions are flipped horizontally, as there is a single
generator for the entire eye region.

Building upon Shuvrajit9904's work:
https://github.com/Shuvrajit9904/PairedCycleGAN-tf/blob/master/parse_face.py
'''

from imutils import face_utils as futil
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

resizeImages = True

mouth_ids = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])

right_eye_region = np.array([17, 18, 19, 20, 21, 39, 40, 41, 36])
left_eye_region = np.array([22, 23, 24, 25, 26, 45, 46, 47, 42])
nose_ids = np.arange(27, 36)
jaw_ids = np.arange(0, 17)

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", mouth_ids),
    ("right", right_eye_region),
    ("left", left_eye_region),
    ("nose", nose_ids),
    ("jaw", jaw_ids)
])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'augment/shape_predictor_68_face_landmarks.dat')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def getDominantColor(img):
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

    return tuple([int(x) for x in centers[0].astype(np.int32)])


def readAndResize(image_path, target_size=512):
    img = cv2.imread(image_path)
    try:
        if (img.size == 0):
            print("The image could not be loaded!")
            return
    except:
        print("Image at " + image_path + " could not be loaded.")
        return

    if (img.shape[1] < img.shape[0]):
        min_dim = img.shape[1]
    else:
        min_dim = img.shape[0]

    if (min_dim > target_size):
        scale = target_size / min_dim

        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))

        img = cv2.resize(img,
                         new_size,
                         interpolation=cv2.INTER_AREA)

        centerY = int(img.shape[0] / 2)
        centerX = int(img.shape[1] / 2)

        if (centerX > centerY):
            rightX = int(centerX + (target_size / 2))
            leftX = int(centerX - (target_size / 2))
            img = img[:, leftX:rightX, :]
        else:
            topY = int(centerY + (target_size / 2))
            bottomY = int(centerY - (target_size / 2))
            img = img[bottomY:topY, :, :]

    return img


def exportImage(status, file_name, part_name, img):
    path_to_exp = "data/parsed/" + status + "/" + file_name + "_" + part_name

    fig = plt.figure(frameon=False)

    plt.axis("off")

    imgplot = plt.imshow(img)
    plt.savefig(path_to_exp, bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close(fig)


def extractFeatures(img, detector, predictor, dominant_color, status, file_name):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray_img, 1)

    if len(rectangles) < 1:
        print("Could not find any faces!")
        return None

    shape = predictor(gray_img, rectangles[0])
    shape = futil.shape_to_np(shape)

    for (name, id_arr) in FACIAL_LANDMARKS_IDXS.items():

        clone = img.copy()

        if (name != "jaw"):
            (x, y), radius = cv2.minEnclosingCircle(np.array([shape[id_arr]]))
            center = (int(x), int(y))
            radius = int(radius) + 20

            mask = np.zeros(clone.shape, dtype=np.uint8)
            mask = cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)

            result_array = (clone & mask)
            y_min = max(0, center[1] - radius)
            x_min = max(0, center[0] - radius)
            result_array = result_array[y_min:center[1] + radius,
                                        x_min:center[0] + radius, :]

            result_array[np.where(
                (result_array == [0, 0, 0]).all(axis=2))] = dominant_color

            if name == 'left':
                exportImage(status, file_name, str(name) + "_eye", cv2.cvtColor(
                    cv2.flip(result_array, 1), cv2.COLOR_BGR2RGB))

            else:
                if name == 'right':
                    name = name + "_eye"
                exportImage(status, file_name, name, cv2.cvtColor(
                    result_array, cv2.COLOR_BGR2RGB))

    return shape


def extractFace(path_to_img, status, file_name, faceCascade, detector, predictor):
    if resizeImages == True:
        img = readAndResize(path_to_img)
    else:
        img = cv2.imread(path_to_img)
        if (img.size == 0):
            print("The image could not be loaded!")
            return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(200, 200)
    )

    if len(faces) < 1:
        print("Could not find a face!")
        return None

    (x, y, w, h) = faces[0]

    copy = img.copy()

    face = copy[y:y + h, x:x + w]

    exportImage(status, file_name, "face",
                cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    dominant_color = getDominantColor(face)

    shapeFeatures = extractFeatures(
        face, detector, predictor, dominant_color, status, file_name)

    if shapeFeatures is None:
        return None

    face_copy = face.copy()

    thresh = cv2.inRange(face_copy, (160, 160, 160), (170, 170, 170))
    face_copy[thresh == 255] = dominant_color

    thresh = cv2.inRange(face_copy, (0, 0, 0), (100, 100, 100))
    face_copy[thresh == 255] = dominant_color

    blur = cv2.blur(face_copy, (10, 10))

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 90, 150, cv2.THRESH_BINARY)

    kernel = np.ones((8, 8), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    face_copy[dilation == 0] = dominant_color

    for (name, id_arr) in FACIAL_LANDMARKS_IDXS.items():
        if (name != "jaw"):
            cv2.fillPoly(face_copy, pts=[
                         shapeFeatures[id_arr]], color=dominant_color)
        else:

            img_shape = face_copy.shape

            jaw_pixels = shapeFeatures[jaw_ids]
            outside_pixels = np.array([[0, img_shape[1]], [0, 0], [15, 0]])

            outside_pixels = np.append(outside_pixels, jaw_pixels, axis=0)

            outside_pixels = np.append(outside_pixels, [
                                       [img_shape[0] - 15, 0], [img_shape[0], 0], [img_shape[0], img_shape[1]]], axis=0)

            cv2.fillPoly(face_copy, pts=[outside_pixels], color=dominant_color)

    exportImage(status, file_name, "skin",
                cv2.cvtColor(face_copy, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":

    for s in ["rug_healthy", "rug_sick", "cfd_healthy", "cfd_sick", "validation_healthy", "validation_sick"]:
        print("Scanning ", s, " patients...")
        for path in os.listdir("data/unparsed/" + s):

            # Skip .gitkeep
            if path.startswith('.'):
                continue

            full_path = os.path.join("data/unparsed/" + s, path)
            if os.path.isfile(full_path):
                split = full_path.split(os.sep)
                status = split[-2]
                file_name = split[-1].split(".")[0]

                print("[INFO] Scanning ", file_name)

                if "cfd" in status:
                    status = "training" + status[3:]

                extractFace(full_path, status, file_name,
                            faceCascade, detector, predictor)
        print("Finished!\n")
