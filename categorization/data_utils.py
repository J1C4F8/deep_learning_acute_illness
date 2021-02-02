import os
import cv2
import numpy as np
import pickle


def load_data(folder_sick, folder_healthy, image_size, ftype, extra_healthy=None, extra_sick=None):
    files_healthy = os.listdir(folder_healthy)
    files_sick = os.listdir(folder_sick)
    data = []
    labels = []

    if extra_healthy is None:
        extra_healthy = ftype
    if extra_sick is None:
        extra_sick = ftype

    for filename in files_healthy:
        sick = np.array([0])
        full_path = folder_healthy + "/" + str(filename)
        if ((ftype in filename) or (extra_healthy in filename)) \
                and os.path.isfile(full_path):
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    for filename in files_sick:
        sick = np.array([1])
        full_path = folder_sick + "/" + str(filename)
        if ((ftype in filename) or (extra_sick in filename)) \
                and os.path.isfile(full_path):
            image = cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=(
                image_size, image_size), interpolation=cv2.INTER_CUBIC)
            data.append(np.asarray(image, dtype=np.int32))
            labels.append(np.asarray(sick, dtype=np.int32))
    return np.asarray(data, dtype=np.float64) / 255, np.asarray(labels, dtype=np.int32)


def make_stacked_sets(image_folder_sick, image_folder_healthy, image_size):
    train_images_mouth, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "mouth")
    train_images_nose, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "nose")
    train_images_skin, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "skin")
    train_images_right_eye, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "_right", extra_sick="eye")

    perm = np.random.permutation(len(train_images_mouth))
    print(len(train_images_mouth), len(train_images_nose), len(train_images_skin), len(train_images_right_eye))
    train_images = [train_images_mouth[perm], train_images_nose[perm],
                    train_images_skin[perm], train_images_right_eye[perm]]
    train_labels = train_labels[perm]
    return np.asarray(train_images), np.asarray(train_labels)


def make_stacked_sets_unshuffled(image_folder_sick, image_folder_healthy, image_size):
    train_images_mouth, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "mouth")
    train_images_nose, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "nose")
    train_images_skin, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "skin")
    train_images_right_eye, train_labels = load_data(
        image_folder_sick, image_folder_healthy, image_size, "_right")

    train_images = [train_images_mouth, train_images_nose,
                    train_images_skin, train_images_right_eye]

    return np.asarray(train_images), np.asarray(train_labels)


def load_shuffled_data(folder_sick, folder_healthy, image_size, ftype):
    data, labels = load_data(folder_sick, folder_healthy, image_size, ftype)
    permutation = np.random.permutation(len(data))
    return data[permutation], labels[permutation]


def save_history(save_path, history, feature, i):
    if i < 3:
        with open(save_path + str(feature) + "/history_" + str(i) + ".pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    else:
        with open(save_path + str(feature) + "/history.pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


def to_labels(predictions):
    pred = np.zeros((len(predictions), 1))
    for i in range(len(predictions)):
        if predictions[i] < 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    return pred


def compute_per_participant(pred, val_labels, folds, feature):
    if feature == 'eye':
        return compute_per_participant_step(pred, val_labels, folds)

    per_participant = np.zeros(len(val_labels))
    for i in range(folds):
        for j in range(0, len(val_labels)):
            if pred[i * 10 + j] == 1 and val_labels[j] == 1:
                per_participant[j] += 1
            if pred[i * 10 + j] == 0 and val_labels[j] == 0:
                per_participant[j] += 1
    return per_participant / folds


def compute_per_participant_step(pred, val_labels, folds):
    per_participant = np.zeros(int(len(val_labels) / 2))
    for i in range(folds):
        for j in range(0, len(val_labels), 2):
            if pred[i * 10 + j] == 1 and val_labels[j] == 1:
                per_participant[int(j / 2)] += 1
            if pred[i * 10 + j] == 0 and val_labels[j] == 0:
                per_participant[int(j / 2)] += 1
    return per_participant / folds
