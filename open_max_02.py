##추가해야할것 나머지 4가지클래스를 클래스5번으로 한번에 묶고 test set과 함께 검증데이터로 사용하여서 결과값 추론

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

#fashion_mnist = tf.keras.datasets.fashion_mnist #fashion mnist 데이터를 호출함
train_X, test_X, train_Y, test_Y = np.load("./image_data_0.npy", allow_pickle=True) # 학습데이터와 라벨링 데이터를 호출한 fashion mnist데이터를 통하여 선언 


class_names = ['crack','pothole','Unknown_set']

# print(len(train_X), len(test_X))

NofClass = 2 #우리가 알고있는 데이터를 몇가지인지 정하는 것을 NofClass라는 변수로 정함 (Fashion mnist기준 라벨링은 10개이므로 5개로 지정)


# 모르는사람에게 설명할 수 있을 정도로 
# (생각까지 적기) openset을 정의하기 위해서 우리가 알고 있는 데이터와 모르는 데이터로 가정을 하기 위한 것
# 알고있는 데이터를 분류모델로 학습시켜서 알고있는 클래스들 뿐만 아니라, 모르는 나머지 open클래스까지
# 추론하는 것을 검증하기 위한 간단한 데이터를 MNIST데이터를 기반으로 만들기 위해서 아래 코드를 작성



#먼저 가지고 있는 학습데이터와 라벨데이터를 아는데이터와 모르는데이터로 분할하기 위하여 이 함수를 작성함
#클래스 순서대로 NofClass 개수만큼 분할
def splitSet(data, GT, NofClass): #분할을 하기위한 splitset함수 선언 입력값은 (학습데이터, 라벨데이터, 분할하고 싶은 데이터 종류(int))
    
    N = GT.shape[0] #shape을 해보면 개수x차원(x ,y)을 N으로 선언 라벨데이터의 개수 
    knownSet_X = [] #우리가 알고 있는 데이터의 개수 리스트를 knownSet_X 라는 변수명으로 선언
    unKnownSet_X = [] #모르고 있는 데이터의 개수 리스트를 unKnownSet_X 라는 변수명으로 선언
    knownSet_Y = [] #알고있는 데이터의 라벨링 데이터 리스트를 knownSet_Y 라는 변수명으로 선언
    unKnownSet_Y = [] # 모르고 있는 데이터의 라벨링 데이터 리스트를 unKnownSet_Y 라는 변수명으로 선언
    print(GT.shape)
    print("GT[0]",GT[0].shape)
    for i in range(N): # N번 만큼 반복
        #print(np.where(GT[i]==1))
        label = np.where(GT[i]==1)[0]
        if label< NofClass: # 만약 GT의 번호가 Nofclass보다 작으면 
            knownSet_X.append(data[i]) #knownSet_X에 학습데이터를 추가
            knownSet_Y.append(label) # knownSet_Y에 라벨데이터를 추가
        else: # 그렇지 않으면
            unKnownSet_X.append(data[i]) #unKnownSet_X에 학습데이터를 추가
            unKnownSet_Y.append(label) # unKnownSet_Y에 라벨데이터를 추가

    knownSet_X = np.array(knownSet_X) #분할한 knownSet_X를 Numpy형식으로 변환
    knownSet_Y = np.array(knownSet_Y)#분할한 knownSet_Y를 Numpy형식으로 변환
    unKnownSet_X = np.array(unKnownSet_X)#분할한 unKnownSet_X를 Numpy형식으로 변환
    unKnownSet_Y = np.array(unKnownSet_Y)#분할한 unKnownSet_Y를 Numpy형식으로 변환
    # print(knownSet_X.shape) # 잘 분할 되었는지 knownSet_X를 출력

    return knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y #함수의 리턴값은 knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y로 지정


#모델에 의해서 predict된 결과 Nx클래스개수 
def clustering(soft, logit): #데이터를 군집화하기 위한 함수 선언 (소프트맥스 결과, logit_vector의 결과)
    Ns = soft.shape[0] # 소프트맥스의 결과물의 갯수를 Ns로 선언
    # print(Ns)
    clustered = [ [], [] ] # 군집화를 하기 위한 리스트를 clustered로 선언 

    for i in range(Ns): #Ns만큼 반복을 한다.(30000번 반복)
        clusterIdx = np.argmax(soft[i]) #0~4 #softmax를 거친후 나온 결과물의 클래스들을 ClusterIdx로 선언 
        clustered[clusterIdx].append(logit[i]) #clustered[clusterIdx]에 로직벡터의 i에 해당하는 값을 추가

    for i in range(NofClass): #NofClass만큼 반복 (5번)
        clustered[i] = np.array(clustered[i]) #clusterde[i]를 numpy형태로 변환
        # print(clustered[i].shape) 

    print("in", len(clustered[0]), len(clustered[1]))
    
    return clustered #리턴 값은 clusterd된 결과물

#임의의 클래스터링 중점벡터
def getMean(data, NofClass): #클러스터링의 중점 벡터를 구하기 위한 함수 선언 (클러스터읠 결과값, 클래스 갯수)
    Nd = data.shape[0] #각 클래스터의 갯수 (x번군집 x개)를 Nd로 선언
    # print("Nd", data.shape)
    S = [0, 0] # 각 벡터의 합을 구하기 위하여 S를 선언 (클래스가 5개라서 5차원으로 선언)

    for i in range(NofClass): #클래스 갯수 만큼 반복
        for j in range(Nd-1): #Nd-1만큼 반복
            S[i] = S[i] + data[j][i]  #S의 i번 만큼 data의 shape(클래스별 총갯수, 클래스갯수)을 더함
    # print(S)
    S = np.array(S) #S를 numpy형식으로 변환
    meanVector = S/Nd #평균을 구하기위하여 S를 Nd로 나눈값을 meanVector로 선언한다

    # print(meanVector)

    return meanVector #리턴값은 평균을 구한 meanVector

#클래스터와 클래스터 간의 거리
def getEuclideanDist(v1, v2, NofClass): #각 클래스의 클래스터 중심값을 구하기 위한 함수 선언 (벡터1,벡터2,클래스갯수)

    S = 0 #중심벡터의 거리를 구하기 위하여 S를 0으로 초기화
    for i in range(NofClass): #각 클래스 만큼 반복
        S = S + (v1[i] - v2[i])*(v1[i] - v2[i]) #두 점 사이의 거리를 구하기 위하여 (v1-v2)를 곱한다.

    dist = math.sqrt(S) #S의 제곱근을 구한다.

    return dist #공식을 대입한 값을 리턴한다.

#get Distances Between Logits And Vector
def getDistances(mean, cluster): #각 클러스터의 중심백터간의 거리를 구하기 위하여 getDistances(평균, 클러스터)를 선언함
    distanceList = [] #distanceList라는 빈 리스트 선언

    for point in cluster: #클러스터의 각 갯수만큼 반복
        dist = getEuclideanDist(mean, point, NofClass) #평균값과, 점수, 클래스갯수를 인풋으로 각 클러스터링의 중간벡터간의 거리를 구한다.
        distanceList.append(dist) #결과물을 distanceList에 추가

    return distanceList #계산한 거리를 리턴한다.


#Extreme value estimation  #각 클래스별로 평균 로짓벡터와의 거리중 가장 큰 값들(20개) input
def fitweibull(x):
   def optfun(theta):
        return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.0001, ftol = 0.0001, disp = 0)


##weibull Distribution

# m = shape factor (형상인자)
# n = scale factor (척도인자)
#PDF(Probability Density Function)
def my_weibull(x,m,n):
    return (m/n) * (x/n)**(m-1)*np.exp(-(x/n)**m)
#CDF(Cumulation Distribution Function)
def my_weibull_cdf(x,m,n):
    return 1-np.exp(-(x/n)**m)


#soft max
def softmax(logits):
  exps = np.exp(logits)
  sumExps = np.sum(exps)
  P = exps / sumExps
  return P
    
    
    
    
#=========training phase=======================================


knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y = splitSet(train_X, train_Y, NofClass) #train_X, train_Y를 NofClass 개수 만큼 분할하여 knownSet_X, knownSet_Y, unKnownSet_X, unKnownSet_Y 선언
# print(train_Y.shape)
# print(len(knownSet_X), len(knownSet_Y), len(unKnownSet_X), len(unKnownSet_Y)) # 잘 분할 되었는지 출력

# print(knownSet_Y[2000])

test_X=np.append(test_X,unKnownSet_X)
test_Y=np.append(test_Y,unKnownSet_Y)

train_X = knownSet_X #knownSet_X를 학습데이터로 선언
train_Y = knownSet_Y #knownSet_Y를 학습라벨링으로 선언

# 0~1 normalize
train_X = train_X / 255.0  #train_X를 정규화 하기위해 255.0으로 나누어 준다
test_X = test_X / 255.0 #test_X 정규화 하기위해 255.0으로 나누어 준다

#reshape the image dimension to (#, 28, 28, 1)
train_X = train_X.reshape(-1, 64, 64, 3) #train_X의 개수만큼 인풋으로 넣어야 하기 때문에 (-1,28,28,1)로 Reshape해준다
test_X = test_X.reshape(-1, 64, 64, 3) #test_X 개수만큼 인풋으로 넣어야 하기 때문에 (-1,28,28,1)로 Reshape해준다



print(train_X.shape)
print(train_Y.shape)


print(train_X.shape, test_X.shape) #train x와 test x의 shape를 출력 

# #모델을 선언 대중적인 이미지 분류모델인 CNN을 사용
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
      #tf.keras.layers.Dense(units=5, activation='softmax')                        
])

#모델을 컴파일함 optimizer는 tf.keras.optimizers.Adam(), loss function은 sparse_categorical_crossentropy를 사용 metrics는 [accuracy]를 사용
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.summary() #모델을 요약(summary)함

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

# 하나띄워보기
# plt.imshow(test_X[36295], cmap='gray')
# plt.colorbar()
# plt.show()

model = load_model('./model/open_max2.h5')

# history = model.fit(train_X, train_Y, epochs=10, validation_split=0.25) #모델을 학습 epochs는 빠른 확인을 위하여 10번 validation_split = 0.25
result_softmax = model.predict(train_X) #모델을 train_X로 검증함 
# print(result_softmax[0]) #첫번째 데이터의 검증 결과를 출력
# print(np.argmax(result_softmax[0])) #첫번째 데이터의 검증결과의 최대값의 위치를 출력(몇번째 클래스인지 확인 0~4)

logit_vector_model = tf.keras.Model(inputs=model.input, outputs=model.layers[13].output) #softmax하기전 데이터 모델의 Logits 벡터를 logict_vector_model로 선언
result_logit = logit_vector_model.predict(train_X) #train_x의 logits vector model검증 결과값을 result_logit으로 선언 
# print(result_logit[0]) #0번째 데이터의 result_logit값을 추출 


clustered = clustering(result_softmax, result_logit) #소프트맥스의 결과값과 로직벡터 값을 군집화한다.(clustered)
# print(len(clustered))

means = [] #clusterde의 중심 벡터를 찾기위하여 평균을 구하기 위해 means를 선언
for i in range(NofClass): #클래수 갯수만큼 반복
    means.append(getMean(clustered[i], NofClass))  #각클래스별 클러스터링결과와, 클래스 갯수를 인풋으로 하여 평균벡터를 구한 후 means에 추가한다.
    
# print(means) #결과를 출력한다.


distLists = []  #각 클래스터의 평균 로직을 구하기 위한 리스트를 선언
for i in range(NofClass): #각 클래스의 갯수만큼 반복
    distList = getDistances(means[i], clustered[i]) #각 클래스터의 평균로직벡터의 거리와 클러스터된 결과물을 인풋으로 거리를 구한다. 
    distList.sort(reverse=True) # 거리를 계산한후 내림차순으로 정렬
    distLists.append(distList[:20]) # 평균 로직벡터와의 거리중 가장 큰 20개의 값을 각 클래스 별로 추출

# print(distLists) #0번째 클래스의 거리 리스트를 출력한다. 
    

weibullParams = []
for i in range(NofClass):
    param = fitweibull(distLists[i])
    # print(param)
    weibullParams.append(param)


# #=========test phase=======================================
open_max=[],[],[]
#어떤 케이스가 들어왔을때
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
# print(class_names[3],"는 총 테스트 데이터",len(test_X),"개 중에",percent[3],"%인",len(open_max[3]),"개 입니다.")
# print(class_names[4],"는 총 테스트 데이터",len(test_X),"개 중에",percent[4],"%인",len(open_max[4]),"개 입니다.")
# print(class_names[5],"는 총 테스트 데이터",len(test_X),"개 중에",percent[5],"%인",len(open_max[5]),"개 입니다.")



print("openmax의 갯수는",len(open_max),"개 입니다.")
