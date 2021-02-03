import os
import sys

import tensorflow as tf
from numpy import interp
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.getcwd())
from categorization.models import make_model
from categorization.plot_utils import *
from categorization.data_utils import *

if __name__ == "__main__":

    image_folder_training_sick = 'data/parsed/training_sick'
    image_folder_training_healthy = 'data/parsed/training_healthy'
    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'

    save_path = 'categorization/model_saves/'
    face_features = ["mouth", "nose", "skin", "eye"]

    base_fpr = np.linspace(0, 1, 101)
    image_size = 128
    folds = 10

    skfold = StratifiedKFold(n_splits=folds, shuffle=False, random_state=1)

    for feature in face_features:
        auc_sum = 0
        tprs = []
        fold_no = 1

        print("[INFO] Training %s" % (feature))

        val_images, val_labels = load_shuffled_data(
            image_folder_val_sick, image_folder_val_healthy, image_size, feature)
        images, labels = load_shuffled_data(
            image_folder_training_sick, image_folder_training_healthy, image_size, feature)

        plt.figure()

        for train, test in skfold.split(images, labels):

            tf.keras.backend.clear_session()
            model = make_model(image_size, feature)

            early_stopping = EarlyStopping(
                monitor='val_F1_metric', mode='max', patience=10, verbose=1)
            model_check = ModelCheckpoint(
                save_path + str(feature) + '/model_' + str(fold_no) + '.h5', monitor='val_F1_metric', mode='max',
                verbose=1, save_best_only=True)

            history = model.fit(images[train], labels[train], epochs=50, batch_size=4,
                                callbacks=[early_stopping, model_check], validation_data=(images[test], labels[test]))

            save_history(save_path, history, feature, fold_no)

            all_saves = os.listdir(save_path + str(feature))
            for save in all_saves:
                # print(save)
                if str(fold_no) + '.h5' in save:
                    best_model_path = save_path + str(feature) + "/" + save

            saved_model = tf.keras.models.load_model(best_model_path, compile=False)
            del model

            if fold_no == 1:
                predictions = to_labels(saved_model.predict(val_images))
            else:
                predictions = np.concatenate(
                    (predictions, to_labels(saved_model.predict(val_images))), axis=0)

            fold_no += 1

            pred = (saved_model.predict(val_images))
            fpr, tpr, _ = roc_curve(val_labels, pred)
            auc_sum += auc(fpr, tpr)
            del saved_model

            plt.plot(fpr, tpr, 'b', alpha=0.15)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        print_roc_curve(tprs, auc_sum, feature, folds)
        print_confusion_matrix(predictions, val_labels, feature, folds)
