# python train_detector_model.py --data dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

EPOCHS = 20
BS = 32

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="path to dataset")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to mask detector model")
args = vars(ap.parse_args())

# initialize the list of data and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["data"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

# convert data and labels to arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# train/test splits using 80/20
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load prebuilt MobileNetV2 network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of model that will be placed on top of base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place head FC model on top of base model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze all layers in base model
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[UPDATE] compiling model...")

# learning rate scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=len(trainX) // BS,
    decay_rate=0.96,
    staircase=True
)
opt = Adam(learning_rate=lr_schedule)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train head of network
print("[UPDATE] training head...")
HEAD = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions
print("[UPDATE] evaluating neural network...")
predIndexes = model.predict(testX, batch_size=BS)

# get index of label with largest predicted proba
predIndexes = np.argmax(predIndexes, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1), predIndexes,
                            target_names=lb.classes_))

# serialize model 
print("[UPDATE] saving mask detector model...")
model.save(args["model"] + ".keras")

# plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, EPOCHS), HEAD.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), HEAD.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, EPOCHS), HEAD.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), HEAD.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plt.savefig('plot.png')
