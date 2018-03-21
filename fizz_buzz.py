from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,LeakyReLU,LSTM
from keras.optimizers import RMSprop,SGD
from keras.utils import np_utils
from keras.callbacks import Callback,EarlyStopping

import numpy

num_digits = 12 # binary encode numbers
nb_classes = 4 # 4 classes : number/fizz/buzz/fizzbuzz
batch_size = 128
timesteps = 8
data_dim =   num_digits

def fb_encode(i):
    if   i % 15 == 0: return [3]
    elif i % 5  == 0: return [2]
    elif i % 3  == 0: return [1]
    else:             return [0]

def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz_pred(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]

def fizz_buzz(i):
    if   i % 15 == 0: return "fizzbuzz"
    elif i % 5  == 0: return "buzz"
    elif i % 3  == 0: return "fizz"
    else:             return str(i)

def create_dataset(x,y):
    dataX,dataY = [],[]
    for i in range(x,y):
         dataX.append(bin_encode(i))
         dataY.append(fb_encode(i))

    return numpy.array(dataX),np_utils.to_categorical(numpy.array(dataY), nb_classes)


dataX,dataY = create_dataset(1,2048)
#testingX,testingY = create_dataset(2000,3000)

model = Sequential()
model.add(Dense(input_dim=num_digits,units=100,activation='relu'))

for x in range(0,16):
    model.add(Dense(units=100, activation='relu'))

model.add(Dense(units=4))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(dataX,dataY,batch_size=30,epochs=1000,validation_split=0.4)

result=model.evaluate(dataX,dataY);#,batch_size=1000)

print('\nAcc' , format( result[1] , '0.2f' ))

