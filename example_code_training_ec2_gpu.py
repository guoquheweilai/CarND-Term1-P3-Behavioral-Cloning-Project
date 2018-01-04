# Local machine environment
# Windows 10 64-bit
#
# AWS environment configuration
# https://github.com/udacity/CarND-Term1-Starter-Kit.git
#
# Simulator settings
# Screen resolution: 1280 x 960
# Graphics quality: Simple
# Select monitor: Display 1 (Left)
# Windowed: Checked
#
# Change log
# 01/02/2018: Use given sample data as training and validation dataset
# 01/02/2018: Add lambda layer
# 01/02/2018: Use LeNet architecture
# 01/03/2018: Switch to AWS
# 01/03/2018: Add augmented data
# 01/03/2018: Use multiple cameras
# 01/03/2018: Add cropping layer
# 01/03/2018: Change from LeNet to NVIDIA architecture
# 01/03/2018: Use generator
# 01/03/2018: Add bgr2rgb function to avoid color space difference between drive.py and cv2.imread
# 01/03/2018: fix multiple cameras bug and add correction
# 01/03/2018: disable multiple cameras
# 01/04/2018: complete the log
# end-of-log

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

samples = []
with open('./recorded_data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def bgr2rgb(bgr_img):
	# Color image loaded by OpenCV is in BGR mode.
	# your model will be fed with RGB images by drive.py
	b,g,r = cv2.split(bgr_img)
	rgb_img = cv2.merge([r,g,b])
	return rgb_img

def sign(x):
    return {
        '0': 0,
        '1': 1,
		'2': -1,
    }.get(x, -100) # -100 is default if x not found
	
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:min(offset+batch_size,num_samples)]
			images = []
			angles = []
			augmented_images, augmented_angles = [], []
			# Tuning this parameter for left and right camera correction
			correction = 0.00625
			for batch_sample in batch_samples:
				for i in range(1): # ==1 Disable left and right camera data as input
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = './recorded_data/data/IMG/' + filename
					## current_path = filename
					#print(current_path)
					image = bgr2rgb(cv2.imread(current_path))
					images.append(image)
					angle = float(batch_sample[3])+sign(i)*correction
					angles.append(angle)
			for image,angle in zip(images, angles):
				augmented_images.append(image)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle)
				augmented_angles.append(angle*(-1.0))
			
			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(ch,row,col), output_shape=(ch,row,col)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=5)

model.save('model.h5')
exit()