# Description
This project aims to detect whether a person is wearking a mask using CNNs. In light of the COVID-19 pandemic, several universities and other places of high traffic mandated the policy of wearing masks. This application aims to aid in enforcing the mask mandate by using CNNs such as MobileNetV2 to classify whether a person is adhering the mask policies. 

First clone the repository to test out the mask detector. You can evaluate the trained neural network based on an example image or try it out on yourself in real-time!

## USAGE:
#### Image Detection
python mask_detect.py --image examples/example_01.png

#### Real-time Video Detection
python mask_detect_video.py