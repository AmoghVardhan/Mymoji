import numpy as np
from keras.layers import Dense
from keras.models import Sequential

data = np.load('Data/images.npy')
label = np.load('Data/labels.npy')

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
# model.evaluate(test_data,test_label)
prediction = model.predict(test_data)

i=0
for x in prediction:
    print(str(x)+' ------------------------ '+str(test_label[i])+'\n')
    i+=1
score = model.evaluate(test_data,test_label)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
