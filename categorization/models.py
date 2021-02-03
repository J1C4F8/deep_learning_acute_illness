from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, FalseNegatives, FalsePositives, TruePositives, TrueNegatives
import tensorflow as tf


def Sensitivity(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


def Specificity(y_true, y_pred):
    true_negatives = tf.keras.backend.sum(
        tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())


def F1_metric(y_true, y_pred):
    sens = Sensitivity(y_true, y_pred)
    spec = Specificity(y_true, y_pred)
    return 2 * ((spec * sens) / (spec + sens + tf.keras.backend.epsilon()))


def make_model(image_size, feature, mcompile=True):
    model = models.Sequential()

    model.add(layers.Conv2D(image_size, (3, 3), padding="same", activation='relu',
                            input_shape=(image_size, image_size, 3),
                            name="input_" + str(feature)))

    model.add(layers.BatchNormalization(name="batch1_" + str(feature)))
    model.add(layers.Conv2D(int(image_size / 2), (3, 3),
                            activation='relu', name="conv1_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch2_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name="max1_" + str(feature)))

    # model.add(layers.Conv2D(int(image_size/4), (3, 3),
    #                         activation='relu', name="conv2_" + str(feature)))
    # model.add(layers.BatchNormalization(name="batch3_" + str(feature)))
    # model.add(layers.MaxPooling2D((2, 2), name="max2_" + str(feature)))

    model.add(layers.Conv2D(int(image_size / 8), (3, 3),
                            activation='relu', name="conv5_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch6_" + str(feature)))
    model.add(layers.MaxPooling2D((2, 2), name="max3_" + str(feature)))

    model.add(layers.Conv2D(int(image_size / 16), (3, 3),
                            activation='relu', name="conv6_" + str(feature)))
    model.add(layers.BatchNormalization(name="batch7_" + str(feature)))
    model.add(layers.AveragePooling2D((2, 2), name="avg1_" + str(feature)))

    model.add(layers.Flatten(name="flatten_" + str(feature)))
    model.add(layers.Dense(48, activation='relu',
                           name="dense1_" + str(feature)))
    model.add(layers.Dropout(0.2, name="dropout1_" + str(feature)))

    model.add(layers.Dense(16, activation='relu',
                           name="dense2_" + str(feature)))
    model.add(layers.Dropout(0.1, name="dropout2_" + str(feature)))

    model.add(layers.Dense(1, activation='sigmoid',
                           name="dense3_" + str(feature)))

    if mcompile:
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss="binary_crossentropy",
                      metrics=['accuracy', AUC(), Specificity, Sensitivity, F1_metric])

    return model


def define_stacked_model(neural_nets, features, trainable=True):
    if trainable == False:
        for model in neural_nets:
            for layer in model.layers:
                layer.trainable = False

    ensemble_visible = [model.input for model in neural_nets]
    ensemble_outputs = [model.layers[14].output for model in neural_nets]  # The final dense layer of size 16.

    merge = layers.concatenate(ensemble_outputs)
    hidden = layers.Dense(32, activation='relu')(merge)
    hidden_drop = layers.Dropout(0.2)(hidden)
    hidden2 = layers.Dense(16, activation='relu')(hidden_drop)
    hidden3 = layers.Dense(4, activation='relu')(hidden2)
    output = layers.Dense(1, activation='sigmoid')(hidden3)
    model = Model(inputs=ensemble_visible, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy', AUC(), Specificity, Sensitivity, F1_metric])

    return model


def load_all_models(save_path, features):
    all_models = list()
    for feature in features:
        # filename = save_path + str(feature) + '/save.h5'
        filename = save_path + str(feature) + '/model.h5'

        model = models.load_model(filename, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss="binary_crossentropy",
                      metrics=['accuracy', AUC(), Specificity, Sensitivity, F1_metric])

        all_models.append(model)
        print('loaded model of ' + str(feature))
    return all_models
