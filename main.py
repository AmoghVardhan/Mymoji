import numpy as np
import imageio
import cv2
from PIL import Image
from LeNet import LeNet
data = np.load('Data/images.npy')
label = np.load('Data/labels.npy')

train_data = data[:9813]
test_data = data[9813:]
train_label = label[:9813]
test_label = label[9813:]
train_data = train_data.reshape(9813,48,48,1)
test_data = test_data.reshape(4205,48,48,1)
obj = None
def trainEvalFunc():
    print('Enter limit for number of epochs:')
    limit = input()
    optEpoch = obj.findOptEpoch(limit)
    model,training,score = obj.train(optEpoch)
    obj.evaluate(model)
    print('Do you want to save the model?(y or n)')
    ans = input()
    if(ans == "y"):
        obj.saveModel(model)

def evalFunc():
    model = obj.loadModel()
    obj.evaluate(model)


dispatch2 = {
    "1": trainEvalFunc,
    "2": evalFunc,
}

def analyzeFunc():
    print('\nChoose operation:')
    print('1.Train Model and evaluate')
    print('2.Evaluate pretrained model')
    print('Enter choice:')
    arg2 = input();
    dispatch2[arg2]()

def readImg():
    # image = imageio.imread('sample.jpeg')
    image = Image.open('sample.jpeg').convert('L')
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image.resize((48, 48), Image.ANTIALIAS)
    return image

def emotionFunc():
    image = readImg()
    image = np.array(image)
    testImg = image.reshape(1,48,48,1)
    model = obj.loadModel()
    emotion = model.predict_classes(testImg)[0]
    print( emotion)


dispatch1 = {
    "1":emotionFunc,
    "2":analyzeFunc,
}

def switchFunct(value):

    return{
        '1': lambda :LeNet(train_data,test_data,train_label,test_label)
    }.get(value)()

print("\n\n\n\n----------Emotion Detection using DL----------")
print("Choose Model:")
print("1.LeNet-5")
print("Enter choice:")

modelChosen = input()
obj = switchFunct(modelChosen)
# obj = x(train_data,test_data,train_label,test_label)

print("\nWhat do you intend to do?")
print("1.Detect Emotion")
print("2.Analyze Model Performance")
print("Enter choice:")
arg1 = input()
dispatch1[arg1]()