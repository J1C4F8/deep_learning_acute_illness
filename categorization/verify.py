import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from pandas import DataFrame
from seaborn import heatmap

sys.path.append(os.getcwd())
from categorization.data_utils import load_data
from categorization.models import *

def get_accuracy(test_labels, prediction_labels, thresh=0.5):
    sum_acc = 0.0
    for i in range(len(test_labels)):
        if (test_labels[i] == (prediction_labels[i] >= thresh)):
            sum_acc += 1
    
    return sum_acc / len(test_labels)

print("Loading data...")

image_size = 128
thresholds = [0.5, 0.6, 0.7, 0.8]
folds = 10

test_faces, _ = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "face")
test_images_mouth, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "mouth")
test_images_face, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "nose")
test_images_skin, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "skin")
test_images_right_eye, test_labels = load_data(
    'data/parsed/validation_sick', 'data/parsed/validation_healthy', image_size, "_right")

test_images = [test_images_mouth, test_images_face, test_images_skin, test_images_right_eye]

print("Loading model and making predictions...")

# for feature in ["mouth", "nose", "skin", "eye", "stacked"]:
#     print("Predicting for " + feature + "...")
#     accs = {
#         0.5: [],
#         0.6: [],
#         0.7: [],
#         0.8: []
#         }
#     for fold_no in range(1,folds+1):
#         model = tf.keras.models.load_model(
#             "categorization/model_saves/" + feature + "/model_" + str(fold_no) + '.h5', compile=False)

#         if feature == "stacked":
#             imgs = test_images
#         elif feature == "mouth":
#             imgs = test_images[0]
#         elif feature == "nose":
#             imgs = test_images[1]
#         elif feature == "skin":
#             imgs = test_images[2]
#         elif feature == "eye":
#             imgs = test_images[3]

#         pred = model.predict(imgs)

#         for thresh in thresholds:
#             acc = get_accuracy(test_labels, pred, thresh)
#             # print("[Threshold {:.2f}] Accuracy fold {:d}: {:.4f}".format(thresh, fold_no, acc))
#             accs[thresh].append(acc)

#     # for thresh in thresholds:
#     #     print("[{}] Mean accuracy on {} folds (threshold={:.2f}): {:.4f}".format(feature.upper(), folds, thresh, accs[thresh]/folds))
#     for thresh in thresholds:
#         curr_acc = np.asarray(accs[thresh])
#         max_idx = np.argmax(curr_acc)
#         print(thresh, "max value at ", max_idx, ": ", curr_acc[max_idx])
#     print("---------------------------------------------------\n")

# Predicting for mouth...
# 0.5 max value at  4 :  0.6842105263157895
# 0.6 max value at  4 :  0.7105263157894737
# 0.7 max value at  4 :  0.7105263157894737
# 0.8 max value at  4 :  0.6842105263157895
# ---------------------------------------------------

# Predicting for nose...
# 0.5 max value at  0 :  0.5526315789473685
# 0.6 max value at  2 :  0.5789473684210527
# 0.7 max value at  2 :  0.631578947368421
# 0.8 max value at  2 :  0.6578947368421053
# ---------------------------------------------------

# Predicting for skin...
# 0.5 max value at  8 :  0.5263157894736842
# 0.6 max value at  8 :  0.5263157894736842
# 0.7 max value at  0 :  0.5
# 0.8 max value at  0 :  0.5
# ---------------------------------------------------

# Predicting for eye...
# 0.5 max value at  0 :  0.6052631578947368
# 0.6 max value at  0 :  0.6052631578947368
# 0.7 max value at  0 :  0.6052631578947368
# 0.8 max value at  0 :  0.5789473684210527
# ---------------------------------------------------

# Predicting for stacked...
# 0.5 max value at  6 :  0.631578947368421
# 0.6 max value at  0 :  0.6578947368421053
# 0.7 max value at  0 :  0.6842105263157895
# 0.8 max value at  0 :  0.7105263157894737
# ---------------------------------------------------

feature = "stacked" # "mouth", "nose", "skin", "eye", "stacked"
fold = 1 # 5, 2, 8, 1, 1
thresh = 0.8 # 0.7, 0.8, 0.6, 0.5, 0.8

if feature == "stacked":
    imgs = test_images
elif feature == "mouth":
    imgs = test_images[0]
elif feature == "nose":
    imgs = test_images[1]
elif feature == "skin":
    imgs = test_images[2]
elif feature == "eye":
    imgs = test_images[3]

model = tf.keras.models.load_model(
            "categorization/model_saves/" + str(feature) + "/model_" + str(fold) + ".h5", compile=False)

pred = model.predict(imgs)

fpr, tpr, _ = roc_curve(test_labels, pred)
auc_sum = auc(fpr, tpr)

plt.plot(fpr, tpr, 'orange')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC Curve for {} model (AUC = {:.3f})".format(str(feature).capitalize(), auc_sum))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.savefig("data/plots/roc_" + str(feature) + "_max.png")

pred = np.int64(pred > thresh)
true = test_labels
print("Acc", get_accuracy(test_labels, pred, thresh))
print(true.reshape(-1))
print(pred.reshape(-1))

matrix = np.zeros((2, 2))
for j in range(len(true)):
    if pred[j] == 1 and true[j] == 1:
        matrix[0][1] += 1
    if pred[j] == 1 and true[j] == 0:
        matrix[1][1] += 1
    if pred[j] == 0 and true[j] == 1:
        matrix[0][0] += 1
    if pred[j] == 0 and true[j] == 0:
        matrix[1][0] += 1

df_cm = DataFrame(matrix, index=["Positives", "Negative"], columns=[
                         "Negative", "Positives"])
plt.figure()
ax = plt.axes()
heatmap(df_cm, annot=True, ax=ax, fmt='g', vmin = 0.0, vmax = 20.0)
ax.set_title('Confusion Matrix ' + str(feature).capitalize())
ax.set_ylabel("Actual Values")
ax.set_xlabel("Predicted Values")
plt.axes().set_aspect('equal', 'datalim')
plt.savefig("data/plots/confusion_matrix_" + str(feature) + "_max.png")
