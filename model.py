import numpy as np
import	matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

data = np.load('Data/images.npy')
label = np.load('Data/labels.npy')

model = Sequential()


#Basic random model
'''
model = Sequential()
model.add(Dense(10,activation='relu',input_shape=(2304,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_data = data[:9813]
test_data = data[9813:]
train_label = label[:9813]
test_label = label[9813:]
train_data = train_data.reshape(9813,2304)
test_data = test_data.reshape(4205,2304)
model.fit(train_data,train_label,validation_split=0.2,epochs=30)
'''


#LeNet-5
model.add(Conv2D(6,kernel_size=5,activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2),padding = 'valid'))
model.add(Conv2D(16,kernel_size=5,activation='relu'))
model.add(MaxPooling2D(pool_size=2,padding = 'valid'))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_data = data[:9813]
test_data = data[9813:]
train_label = label[:9813]
test_label = label[9813:]
train_data = train_data.reshape(9813,48,48,1)
test_data = test_data.reshape(4205,48,48,1)
training = model.fit(train_data,train_label,validation_split=0.2,epochs=30)
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.show()





# model.evaluate(test_data,test_label)
prediction = model.predict(test_data)

i=0
for x in prediction:
    print(str(x)+' ------------------------ '+str(test_label[i])+'\n')
    i+=1
score = model.evaluate(test_data,test_label)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
