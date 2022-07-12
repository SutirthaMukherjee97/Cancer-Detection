gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
from tensorflow.keras.layers import Input,Flatten,Dense,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from glob import glob
base_dir = "/content/SiPakMed"
train_dir = "/content/SiPakMed/train"
test_dir = "/content/SiPakMed/test"
!unrar x -Y "/content/gdrive/MyDrive/SiPakMed.rar"
def arr(y):
    two_d=np.zeros((len(y),5))
    for i in range(len(two_d)):
        two_d[i][y[i]]=1
    return two_d
import pandas as pd
from matplotlib.pyplot import imread
uniques = ["Dyskeratotic" , "Koilocytotic" , "Metaplastic" , "Parabasal" , "SuperficialIntermediate"]
dirs = ["train"]
data = []
for dir in dirs :
  for unique in uniques:
    directory = "/content/SiPakMed/" + dir + "/" + unique

    for filename in os.listdir(directory):
      path = directory + "/" + filename 
      data.append([ filename , path  , unique])
df = pd.DataFrame(data, columns = ["filename" ,"path", "class"])
y = np.array([i for i in df["class"]])
x = np.array([i for i in df["path"]])
def encode_y(y):
  Y = []
  for i in y : 
    if(i == "Dyskeratotic" ):
      Y.append(0)
    elif(i == "Koilocytotic" ):
      Y.append(1)
    if(i == "Metaplastic" ):
      Y.append(2)
    if(i == "Parabasal" ):
      Y.append(3)
    if(i == "SuperficialIntermediate" ):
      Y.append(4)
      
  return  np.array(Y).astype("float32")          

# convert file paths info nums 
#then normalize  
def process_x(x):
   return np.array([imread(i) for i in x ]).astype("float32") / 255.0
X_train = process_x(x)
y_train = encode_y(y)
y_train = y_train.reshape(-1,1)
from tensorflow.python.keras.backend import dtype
def arr(y):
    two_d=np.zeros((len(y),5), dtype=int)
    for i in range(len(two_d)):
        two_d[i][y[i]]=1
    return two_d
y_train = arr(y_train)
import pandas as pd
from matplotlib.pyplot import imread
uniques = ["Dyskeratotic" , "Koilocytotic" , "Metaplastic" , "Parabasal" , "SuperficialIntermediate"]
dirs = ["test"]
data = []
for dir in dirs :
  for unique in uniques:
    directory = "/content/SiPakMed/" + dir + "/" + unique

    for filename in os.listdir(directory):
      path = directory + "/" + filename 
      data.append([ filename , path  , unique])
df = pd.DataFrame(data, columns = ["filename" ,"path", "class"])
y = np.array([i for i in df["class"]])
x = np.array([i for i in df["path"]])
def encode_y(y):
  Y = []
  for i in y : 
    if(i == "Dyskeratotic" ):
      Y.append(0)
    elif(i == "Koilocytotic" ):
      Y.append(1)
    if(i == "Metaplastic" ):
      Y.append(2)
    if(i == "Parabasal" ):
      Y.append(3)
    if(i == "SuperficialIntermediate" ):
      Y.append(4)
      
  return  np.array(Y).astype("float32")          

# convert file paths info nums 
#then normalize  
def process_x(x):
   return np.array([imread(i) for i in x ]).astype("float32") / 255.0
X_test = process_x(x)
y_test = encode_y(y)
y_test = y_test.reshape(-1,1)
y_test = y_test.astype(int)
y_test = arr(y_test)
y_test = y_test.astype(float)
image_size = (256,256)
batch_size = 16
train_datagen = ImageDataGenerator( rotation_range=40,
                                    width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    #shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='reflect',
                                    )
val_datagen = ImageDataGenerator()

train =  train_datagen.flow(X_train, y_train, 
                            batch_size=batch_size,
                            shuffle = True)
validation = val_datagen.flow(X_test ,y_test,
                          batch_size=batch_size,
                          shuffle = True)
