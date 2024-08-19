# python mask_detect.py --image examples/example_01.png

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import cv2
import os

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
args = vars(ap.parse_args())

# load serialized face detector model
print("[UPDATE] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask detector model
print("[UPDATE] loading face mask detector model...")
model = load_model(args["model"])

# load input image, clone it, and grab image spatial dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pass blob through network and obtain face structure detections
print("[UPDATE] computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	# get confidence associated with detection
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring confidence is greater than threshold
	if confidence > 0.6:
		# compute the bounding box for object
		bounded_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = bounded_box.astype("int")

		# ensure bounding boxes fall within dimensions of frame
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		(startX, startY) = (max(0, startX), max(0, startY))



		# extract face, convert it from BGR to RGB channel, resize it to 224x224
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))

		# preprocess face image
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass face through model to determine classification
		(mask, noMask) = model.predict(face)[0]

		# determine class label and color with confidence probability
		label = "Mask" if mask > noMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, noMask) * 100)

		# display label and bounding box rectangle
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


cv2.imshow("Output", image)
cv2.waitKey(0)