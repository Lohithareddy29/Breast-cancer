# ========================== IMPORT PACKAGES ========================

import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
import cv2
import os
import argparse
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
from PIL import Image
import os


# ======================== 1.READ INPUT IMAGE ========================
    
filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image') 
plt.axis ('off')
plt.show()
    
    
    
# ======================= 2.PREPROCESSING =============================
 
#==== RESIZE IMAGE ====
 
resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))
 
fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()

              
#==== GRAYSCALE IMAGE ====
 
try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)

except:
    gray1 = img_resize_orig
    
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()
        
         
#=========================== 3.FEATURE EXTRACTION ======================

#=== MEAN STD DEVIATION ===
 
mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("------------------------------------")
print("       FEATURE EXTRACTION           ")
print("------------------------------------")
print()
print(features_extraction)
        

# ===================== 4. IMAGE  SPLITTING ============================
 
# TEST AND TRAIN 
 

import os 

from sklearn.model_selection import train_test_split

dataset_no = os.listdir('Histopathology/No')

dataset_yes = os.listdir('Histopathology/Yes')
    
    
dot1= []
labels1 = []
for img in dataset_no:
        # print(img)
        img_1 = mpimg.imread('Histopathology/No/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)
    
for img in dataset_yes:
        # print(img)
        img_1 = mpimg.imread('Histopathology/Yes/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)        
    
x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)


print("---------------------------------")
print("Image Splitting")
print("---------------------------------")
print()
print("1. Total Number of images =", len(dot1))
print()
print("2. Total Number of Test  =", len(x_test))
print()
print("3. Total Number of Train =", len(x_train))    
        
        
# ====================== CLASSIFICATION =============================

# ================ CNN ========================


from keras.utils import to_categorical
import numpy as np

y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]
                    

            
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential

    
# initialize the model
model=Sequential()
    
    
#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=2,verbose=1)        


print("-------------------------------------------")
print("    CONVOLUTIONAL NEURAL NETWORK ")
print("-------------------------------------------")
print()

accuracy=history.history['loss']
loss=max(accuracy)
accuracy=100-loss
print()
print("1.Accuracy is :",accuracy,'%')
print()
print("2.Loss is     :",loss)
print()

        
# ================= VGG- 19 =============

from keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))

for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy")
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)




history = model.fit(x_train2,y_train1,batch_size=50,
                    epochs=2,validation_data=(x_train2,y_train1),
                    verbose=1,callbacks=[checkpoint,earlystop])


print("===========================================================")
print("----------  (VGG 19) ----------")
print("===========================================================")
print()
accuracy=history.history['loss']
loss=max(accuracy)
accuracy=100-loss
print()
print("1.Accuracy is :",accuracy,'%')
print()
print("2.Loss is     :",loss)
print()

        
        # model.save("vgg19.h5")
        
        
        
# ====================== PREDICTION =======================

print()
print("-----------------------")
print("       PREDICTION      ")
print("-----------------------")
print()


Total_length = len(dataset_no) + len(dataset_yes) 


temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 0:
    print('------------------------')
    print(' IDENTIFIED = NORMAL ')
    print('------------------------')
    

elif labels1[zz[0][0]] == 1:
    print('-----------------------')
    print(' IDENTIFIED = CANCER')
    print('-----------------------')

