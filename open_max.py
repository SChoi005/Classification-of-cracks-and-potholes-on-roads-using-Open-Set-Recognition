from tkinter import NS
from unittest import result
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import exponweib
from scipy.optimize import fmin
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import os

train_X, test_X, train_Y, test_Y = np.load("./image_data_0.npy", allow_pickle=True) 

class_names = ['crack','pothole','Unknown_set']

# print(len(train_X), len(test_X))

NofClass = 2 

def splitSet(data, GT, NofClass): 
    
    N = GT.shape[0] 
    knownSet_X = []
    unKnownSet_X = []
    knownSet_Y = [] 
    unKnownSet_Y = []
    print(GT.shape)
    print("GT[0]",GT[0].shape)
    for i in range(N):
        #print(np.where(GT[i]==1))
        label = np.where(GT[i]==1)[0]
        if label< NofClass:
            knownSet_X.append(data[i])
            knownSet_Y.append(label)
        else: 
            unKnownSet_X.append(data[i]) 
            unKnownSet_Y.append(label) 

    knownSet_X = np.array(knownSet_X)
    knownSet_Y = np.array(knownSet_Y)
    unKnownSet_X = np.array(unKnownSet_X)
    unKnownSet_Y = np.array(unKnownSet_Y)
    # print(knownSet_X.shape)

    return knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y 


def clustering(soft, logit):
    Ns = soft.shape[0]
    # print(Ns)
    clustered = [ [], [] ] 

    for i in range(Ns):
        clusterIdx = np.argmax(soft[i]) 
        clustered[clusterIdx].append(logit[i]) 

    for i in range(NofClass): 
        clustered[i] = np.array(clustered[i])
        # print(clustered[i].shape) 

    print("in", len(clustered[0]), len(clustered[1]))
    
    return clustered 

def getMean(data, NofClass):
    Nd = data.shape[0] 
    # print("Nd", data.shape)
    S = [0, 0]

    for i in range(NofClass): 
        for j in range(Nd-1): 
            S[i] = S[i] + data[j][i]
    # print(S)
    S = np.array(S) 
    meanVector = S/Nd

    # print(meanVector)

    return meanVector

def getEuclideanDist(v1, v2, NofClass): 

    S = 0 
    for i in range(NofClass):
        S = S + (v1[i] - v2[i])*(v1[i] - v2[i]) 

    dist = math.sqrt(S) 

    return dist

def getDistances(mean, cluster):
    distanceList = []

    for point in cluster: 
        dist = getEuclideanDist(mean, point, NofClass) 
        distanceList.append(dist) 

    return distanceList


def fitweibull(x):
   def optfun(theta):
        return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.0001, ftol = 0.0001, disp = 0)



def my_weibull(x,m,n):
    return (m/n) * (x/n)**(m-1)*np.exp(-(x/n)**m)
def my_weibull_cdf(x,m,n):
    return 1-np.exp(-(x/n)**m)


def softmax(logits):
  exps = np.exp(logits)
  sumExps = np.sum(exps)
  P = exps / sumExps
  return P
    
    
    
    
#=========training phase=======================================


knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y = splitSet(train_X, train_Y, NofClass) 
# print(train_Y.shape)
# print(len(knownSet_X), len(knownSet_Y), len(unKnownSet_X), len(unKnownSet_Y)) 

# print(knownSet_Y[2000])

test_X=np.append(test_X,unKnownSet_X)
test_Y=np.append(test_Y,unKnownSet_Y)

train_X = knownSet_X
train_Y = knownSet_Y 

# 0~1 normalize
train_X = train_X / 255.0 
test_X = test_X / 255.0 

train_X = train_X.reshape(-1, 64, 64, 3)
test_X = test_X.reshape(-1, 64, 64, 3) 



print(train_X.shape)
print(train_Y.shape)


print(train_X.shape, test_X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(64,64,3), kernel_size=(3, 3), filters = 32, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters = 64, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(rate=0.5),  
    tf.keras.layers.MaxPool2D(strides=(2, 2)),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters = 128, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters = 256, padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=NofClass),   
    tf.keras.layers.Softmax()
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.summary() 

model_dir = "./model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
    
model_path = model_dir + "open_max2.h5"
checkpoint = ModelCheckpoint(filepath=model_path , monitor='accuracy', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='accuracy', patience=6)


# std = (int)(len(train_X)*0.2)
# val_X = train_X[:std]
# val_Y = train_X[:std]

history = model.fit(train_X, train_Y, batch_size=32, epochs=100,
                    callbacks=[checkpoint,early_stopping])

# plt.imshow(test_X[36295], cmap='gray')
# plt.colorbar()
# plt.show()

model = load_model('./model/open_max2.h5')

# history = model.fit(train_X, train_Y, epochs=10, validation_split=0.25)
result_softmax = model.predict(train_X)
# print(result_softmax[0]) 
# print(np.argmax(result_softmax[0]))

logit_vector_model = tf.keras.Model(inputs=model.input, outputs=model.layers[13].output)
result_logit = logit_vector_model.predict(train_X) 
# print(result_logit[0]) 


clustered = clustering(result_softmax, result_logit)
# print(len(clustered))

means = [] 
for i in range(NofClass): 
    means.append(getMean(clustered[i], NofClass))
    
# print(means)


distLists = []  
for i in range(NofClass):
    distList = getDistances(means[i], clustered[i])
    distList.sort(reverse=True)
    distLists.append(distList[:20]) 

# print(distLists)
    

weibullParams = []
for i in range(NofClass):
    param = fitweibull(distLists[i])
    # print(param)
    weibullParams.append(param)


# #=========test phase=======================================
open_max=[],[],[]
result_logit = logit_vector_model.predict(test_X)
# print(len(result_logit))
accurate = 0
crack_tp = 0
crack_fp = 0
pothole_tp = 0
pothole_fp = 0
unknown_tp = 0
unknown_fp = 0
for j in range(len(result_logit)):
    # print(np.argmax(result_logit[i]))
    case = result_logit[j]
    weight = []
    for i in range(NofClass):

        dist = getEuclideanDist(case, means[i], NofClass)
        w = my_weibull_cdf(dist, weibullParams[i][0], weibullParams[i][1])
        weight.append(w)
        # print("result_ligit: ",case[i])
        # print("dist, cdf: ", dist, w)
        
    v0 = []
    for i in range(NofClass):
        v0.append(case[i]*weight[i])
    # print(v0)

    openLogits = []
    for i in range(NofClass):
        openLogits.append(case[i]-v0[i])
    # print(openLogits)
    openLogits.append(sum(v0))
    
    # print(openLogits)

    openLogits = np.array(openLogits)

    openMax = softmax(openLogits)    

    openmax_class = np.argmax(openMax)
    if openmax_class == 0:
        open_max[0].append(openmax_class)
    elif openmax_class == 1:
        open_max[1].append(openmax_class)
    elif openmax_class == 2:
        open_max[2].append(openmax_class)

    if openmax_class == (int)(test_Y[j]):
        accurate +=1
        if openmax_class == 0:
            crack_tp+=1
        elif openmax_class == 1:
            pothole_tp+=1
        elif openmax_class == 2:
            unknown_tp+=1
    else:
        if openmax_class == 0:
            crack_fp+=1
        elif openmax_class == 1:
            pothole_fp+=1
        elif openmax_class == 2:
            unknown_fp+=1
    
    print("테스트",j+1,"번째이미지의 결과는",openmax_class,"번째 클래스인 ",class_names[openmax_class],"입니다.", ", 정답 클래스 : ", class_names[(int)(test_Y[j])])
# print(len(open_max[0]))
# print(type(len(open_max[0])))
# print(type(len(test_X)))
# print(openMax)
percent = [0,0,0]
percent[0] ="{:.2f}".format(len(open_max[0])/(len(test_X))*100)
percent[1] = "{:.2f}".format(len(open_max[1])/(len(test_X))*100)
percent[2] = "{:.2f}".format(len(open_max[2])/(len(test_X))*100)
print("정확도 : {:.2f}%".format(accurate/len(result_logit)*100))
print("=================\nPrecison\n-----------------")
print("Crack : {:.2f}%".format(crack_tp/(crack_tp+crack_fp)*100))
print("Pothole : {:.2f}%".format(pothole_tp/(pothole_tp+pothole_fp)*100))
print("unknown : {:.2f}%".format(unknown_tp/(unknown_tp+unknown_fp)*100))
print("=================")

print(class_names[0],"는 총 테스트 데이터",len(test_X),"개 중에",percent[0],"%인",len(open_max[0]),"개 입니다.")
print(class_names[1],"는 총 테스트 데이터",len(test_X),"개 중에",percent[1],"%인",len(open_max[1]),"개 입니다.")
print(class_names[2],"는 총 테스트 데이터",len(test_X),"개 중에",percent[2],"%인",len(open_max[2]),"개 입니다.")



print("openmax의 갯수는",len(open_max),"개 입니다.")
