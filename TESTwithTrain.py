from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
import os

IMAGE_SIZE = 256
IMAGE_DIRECTORY = 'V:/Cat_Or_Not/data/testWithTrain'

#def label_img(name):
#    if name == 'cats': return np.array([1, 0])
#    elif name == 'notcats' : return np.array([0, 1])


def load_data():
  print("Loading images...")
  test_data = []
  directories = next(os.walk(IMAGE_DIRECTORY))[1]

  for dirname in directories:
    print("Loading {0}".format(dirname))
    file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2]
    for i in range(len(file_names)):
      image_name = choice(file_names)
      image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
      if image_name != ".DS_Store":
#        label = label_img(dirname)
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
#        test_data.append([np.array(img), label])
        test_data.append(np.array(img))
  
  return test_data

test_data = np.array(load_data())
test_images = np.array([i for i in test_data]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)

print('Loading model...')
model = load_model("modelProject.h5")

print('Making Prediction...')
# make a prediction
ynew = model.predict_classes(test_images, verbose=1)
# show the inputs and predicted outputs
for i in range(len(test_data)):
  if ynew[i] == 0:
    print("Predicted= Cat")
  else:
    print("Predicted= Not Cat")