import os
import sys

from numpy import interp
from sklearn.metrics import auc, confusion_matrix
from sklearn.metrics import roc_curve

sys.path.append(os.getcwd())
from categorization.models import *
from categorization.plot_utils import *
from categorization.data_utils import *

if __name__ == "__main__":

    # use argument stacked if you only plot the stacked graphs
    # use "all" to plot the graphs for all models
    # use "features" to only plot the graphs for the individual models (without stacked)
    if sys.argv[1] == 'stacked':
        face_features = ['stacked']
    elif sys.argv[1] == 'all':
        face_features = ['mouth', 'nose', 'skin', 'eye', 'stacked']
    elif sys.argv[1] == 'features':
        face_features = ['skin']

    image_folder_val_sick = 'data/parsed/validation_sick'
    image_folder_val_healthy = 'data/parsed/validation_healthy'
    save_path = 'categorization/model_saves/'

    base_fpr = np.linspace(0, 1, 101)
    image_size = 128
    folds = 10

    per_participant = np.zeros((len(face_features), 38))
    auc_values_all = []
    for feature in face_features:

        auc_values = np.zeros(folds)
        tprs = []

        if feature == 'stacked':
            stacked = 1
        else:
            stacked = 0

        if feature == 'eye':
            ftype = 'right_eye'
        else:
            ftype = feature

        print("[INFO] Making plots for %s" % (feature))

        if stacked == 0:
            val_images, val_labels = load_data(image_folder_val_sick, image_folder_val_healthy, image_size, ftype)
        else:
            val_images, val_labels = make_stacked_sets_unshuffled(image_folder_val_sick, image_folder_val_healthy,
                                                                  image_size)

        plt.figure()

        for fold_no in range(1, folds + 1):
            tf.keras.backend.clear_session()
            best_model_path = save_path + str(feature) + "/model_" + str(fold_no) + '.h5'
            saved_model = tf.keras.models.load_model(best_model_path, compile=False)
            saved_model.compile(optimizer=Adam(learning_rate=0.001),
                                loss="binary_crossentropy",
                                metrics=['accuracy', AUC(), Specificity, Sensitivity, F1_metric])

            if stacked == 0:
                pred = (saved_model.predict(val_images))

            else:
                pred = saved_model.predict([val_images[0], val_images[1], val_images[2], val_images[3]])
            if fold_no == 1:
                predictions = to_labels(pred)
            else:
                predictions = np.concatenate((predictions, to_labels(pred)), axis=0)

            fpr, tpr, _ = roc_curve(val_labels, pred)
            auc_values[fold_no - 1] = auc(fpr, tpr)
            del saved_model

            plt.plot(fpr, tpr, 'b', alpha=0.15)
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)

        print_roc_curve(tprs, np.sum(auc_values), feature, folds)
        print_confidence_intervals(predictions, val_labels, auc_values, feature, folds)
        print(predictions.shape, val_labels.shape)
        auc_values_all.append(auc_values)

    plt.figure()
    plt.boxplot(auc_values_all, labels=[feature.capitalize() for feature in face_features], showfliers=False)
    plt.ylabel('AUC scores')
    plt.savefig('data/plots/box_whiskers_plot.png', dpi=300, bbox_inches='tight')
    plt.close()