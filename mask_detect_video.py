# python mask_detect_video.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def mask_detection(frame, mask_model, face_model):
	# construct a blob from image
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass blob through network and obtain face structure detections
	face_model.setInput(blob)
	detections = face_model.forward()

	# initialize list of faces, corresponding locations, and list of predictions from face mask neural network
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# get confidence associated with detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring confidence is greater than threshold
		if confidence > 0.6:
			# compute the bounding box for object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure bounding boxes fall within dimensions of frame
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			(startX, startY) = (max(0, startX), max(0, startY))
			


			# extract face, convert it from BGR to RGB channel, resize it to 224x224
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))

			# preprocess face image
			face = img_to_array(face)
			face = preprocess_input(face)
			# face = np.expand_dims(face, axis=0)

			# append face and bounding boxes to lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# For faster inference, make batch predictions on faces at the same time rather than sequentially in above loop
		faces = np.array(faces, dtype="float32")
		preds = mask_model.predict(faces, batch_size=32)

	# return face locations and prediction locations
	return (locs, preds)

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector_model",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model.keras",
	help="path to trained face mask detector model")
args = vars(ap.parse_args())

# load serialized face detector model
print("[UPDATE] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load face mask detector model
print("[UPDATE] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize video stream
print("[UPDATE] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	# resize frame from video stream
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# pass frame through model to determine classification
	(locs, preds) = mask_detection(frame, maskNet,  faceNet)

	# loop over detections and corresponding locations
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, noMask) = pred

		# determine class label and color with confidence probability
		label = "Mask" if mask > noMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, noMask) * 100)

		# display label and bounding box rectangle
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()