#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np


# In[24]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[25]:


TrainData="./Face Mask Dataset/Train"


# In[26]:


TestData="./Face Mask Dataset/Test"


# In[27]:


ts=(32,32)
bs=32


# In[28]:


#Preprocessing the training dataset
train_gen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True, rotation_range=30,
                                            width_shift_range=0.2,height_shift_range=0.2,fill_mode='nearest')
train_set = train_gen.flow_from_directory(TrainData,
                                             target_size = ts,
                                             batch_size = bs,
                                             interpolation="nearest",
                                             class_mode = 'binary')


# In[29]:


#preprocessing test dataset
test_gen = ImageDataGenerator(rescale = 1./255)
test_set = test_gen.flow_from_directory(TestData,
                                            target_size = ts,
                                            batch_size = bs,
                                            class_mode = 'binary')


# In[30]:


#Building CNN


# In[31]:


cnn = tf.keras.models.Sequential()#initializing cnn


# In[32]:


cnn.add(tf.keras.layers.Conv2D(filters=200, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))#Convolution layer


# In[33]:


cnn.add(tf.keras.layers.AveragePooling2D(pool_size=2, strides=1))#pooling layer


# In[34]:


#second convpolution layer
cnn.add(tf.keras.layers.Conv2D(filters=100, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.AveragePooling2D(pool_size=2, strides=1))


# In[35]:


cnn.add(tf.keras.layers.Flatten())#Flattening


# In[36]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))#fully connected cnn(128)


# In[37]:


cnn.add(tf.keras.layers.Dropout(0.5))#Regularization


# In[38]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))#o/p layer


# In[39]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#compiling


# In[40]:


from keras.callbacks import TensorBoard, ModelCheckpoint
checkpoint = ModelCheckpoint('Model1-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
#to save the model
model=cnn.fit(x = train_set, validation_data = test_set, epochs = 10, callbacks=[checkpoint])#training and testing cnn


# In[21]:


import matplotlib.pyplot as plt
N = 10#epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), model.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), model.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), model.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), model.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch No")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()


# In[22]:


train_set.class_indices


# In[23]:


test_set.class_indices


# In[24]:


from keras.preprocessing import image 
img = image.load_img('./Face Mask Dataset/Single_valid/NM.png', target_size = ts,interpolation="nearest")
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'


# In[25]:


print(prediction)


# In[26]:


from keras.preprocessing import image
img = image.load_img('./Face Mask Dataset/Single_valid/M.png', target_size = ts,interpolation="nearest")
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'


# In[27]:


print(prediction)


# In[28]:


from keras.preprocessing import image
img = image.load_img('./Face Mask Dataset/Single_valid/2.1.jpg', target_size = ts)
#img=ImageDataGenerator(rescale = 1./255)
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
print(img.shape)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'
print(prediction)


# In[32]:


from keras.preprocessing import image
img = image.load_img('./Face Mask Dataset/Single_valid/2.jpg', target_size = ts)
#img=ImageDataGenerator(rescale = 1./255)
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
print(img.shape)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'
print(prediction)


# In[33]:


from keras.preprocessing import image
img = image.load_img('./Face Mask Dataset/Single_valid/22.jpg', target_size = ts)
#img=ImageDataGenerator(rescale = 1./255)
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
print(img.shape)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'
print(prediction)


# In[2]:


import numpy as np
ts=(64,64)
from keras.preprocessing import image
img = image.load_img('./Face Mask Dataset/Single_valid/1.2.jpg', target_size = ts)
#img=ImageDataGenerator(rescale = 1./255)
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
print(img.shape)
result = cnn.predict(img)
if result[0][0] == 0:
    prediction = 'Mask'
else:
    prediction = 'No Mask'
print(prediction)


# In[ ]:




