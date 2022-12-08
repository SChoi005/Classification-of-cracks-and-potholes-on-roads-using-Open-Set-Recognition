from PIL import Image
import cv2
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import libmr


caltech_dir = './dataset/Train'
categories = ['crack', 'pothole', 'unknown']
nb_classes = len(categories)

image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
y = []

# for idx, cat in enumerate(categories):
    
#     #one-hot 돌리기.
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1

#     image_dir = caltech_dir + "/" + cat
#     files = glob.glob(image_dir+"/*.png")
#     print(cat, " 파일 길이 : ", len(files))
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)

#         X.append(data)
#         y.append(label)

#         if i % 700 == 0:
#             print(cat, " : ", f)

# X = np.array(X)
# y = np.array(y)
# #1 0 0 0 이면 airplanes
# #0 1 0 0 이면 buddha 이런식


# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# X_train, X_test, y_train, y_test = train_test_split(X, y)
# xy = (X_train, X_test, y_train, y_test)
# np.save("./image_data_0.npy", xy)

# print("ok", len(y))


X_train, X_test, y_train, y_test = np.load("./image_data_3.npy",allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])


X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255



X_train = X_train.reshape(-1, 64, 64, 3)
def Model():
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
        tf.keras.layers.Dense(units=2, activation='softmax')                        
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

    return model

model = Model()
model_dir = "./model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
model_path = model_dir + "multi_image_classification.h5"
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint])

model = load_model('./model/multi_image_classification.h5')

print("테스트 정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))


y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()




test_img_dir = "./predict_image"
image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(test_img_dir+"/*.png")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
model = load_model('./model/multi_image_classification.h5')
model.summary()
prediction = model.predict(X)


for i, p in enumerate(prediction):
    print('========================')
    print("File name : "+files[i])
    print(p)
    print(categories[np.argmax(p)])
    print('========================')

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

# for i in range (len(X)):
    

#########################################
#### open max 구축###

# # 각 클래스 별로 선별된 데이터를 이용하여 logit vecotr 계산(softmax 들어가기 전 단계)
# model = load_model('./model/multi_image_classification.h5')
# def get_model_outputs(dataset, prob=False):
#     pred_scores = []
#     for x in dataset:
#         model_outputs = model(x, training=False)
#         if prob:
#             model_outputs = tf.nn.softmax(model_outputs)
#         pred_scores.append(model_outputs.numpy())
#     pred_scores = np.concatenate(pred_scores, axis=0)
#     return pred_scores

# train_data = tf.data.Dataset.from_tensor_slices(X_train).batch(32)
# train_pred_scores = get_model_outputs(train_data, False)
# train_pred_simple = np.argmax(train_pred_scores, axis=1)
# print(accuracy_score(y_train, train_pred_simple))

# train_correct_actvec = train_pred_scores[np.where(y_train == train_pred_simple)[0]]
# train_correct_labels = y_train[np.where(y_train == train_pred_simple)[0]]

# # 1)각 클래스 별 activation vector의 평균 계산(class_mean)
# # 2)각 클래스 별 평균 actvec와의 거리 계산(dist_to_means)
# # 3)libmr패키지를 이용해 weibulll distribution fitting
# #    -> eta개의 가장 먼 거리들의 matrix를 극단치라고 정의 (=가장큰값을 뽑으면 가장큰값보다 클 확률은 w분포 만족)
# # 4)각 클래스 별로 거리가 가장 큰 n개의 샘플로 최대 가능도 추정을 통해 극단치분포의 파라미터 추정, 
# #    클래스별 극단치 분포 생성(class_means)
# dist_to_means = []
# mr_models, class_means = [], []
# eta = 100
# for c in np.unique(y_train):
#     class_act_vec = train_correct_actvec[np.where(train_correct_labels == c)[0], :]
#     class_mean = class_act_vec.mean(axis=0)
#     dist_to_mean = np.square(class_act_vec - class_mean).sum(axis=1)
#     dist_to_mean = np.sort(dist_to_mean).astype(np.float64)
#     dist_to_means.append(dist_to_mean)
#     mr = libmr.MR()
#     mr.fit_high(dist_to_mean[-eta:], eta)
#     class_means.append(class_mean)
#     mr_models.append(mr)

# class_means = np.array(class_means)

# # compute_openmax ~ make_prediction
# # 5)새로운 데이터 넣고 actvec를 계산. 기존 클래스의 평균 actvec와 거리 게산
# # 6)극단 분포의 cdf값을 이용해 actvec 업데이트
# def compute_openmax(actvec):
#     #새로운 테스트 데이터가 입력-> 각 클래스별로 테스트 데이터의 actvec와 해당 클래스의 평균actvec가 떨어진 거리
#     dist_to_mean = np.square(actvec - class_means).sum(axis=1).astype(np.float64)
#     scores = []
#     #극단분포의 cdf값(극단값일 확률) 구하기 -> actvec 업데이트
#     #cdf값 작을 수록 클래스별 평균 actvec과 먼 것
#     for dist, mr in zip(dist_to_mean, mr_models):
#         scores.append(mr.w_score(dist))
#     scores = np.array(scores)
#     w = 1 - scores
#     rev_actvec = np.concatenate([
#         w * actvec,
#         [((1 - w) * actvec).sum()]])
#     return np.exp(rev_actvec) / np.exp(rev_actvec).sum()

# def make_prediction(_scores, _T, thresholding=True):
#     _scores = np.array([compute_openmax(x) for x in _scores])
#     if thresholding:
#         uncertain_idx = np.where(np.max(_scores, axis=1) < _T)[0]
#         uncertain_vec = np.zeros((len(uncertain_idx), m + 1))
#         uncertain_vec[:, -1] = 1
#         _scores[uncertain_idx] = uncertain_vec
#     _labels = np.argmax(_scores, 1)
#     return _labels


# thresholding = True

# threshold = 0.7
# test_data = tf.data.Dataset.from_tensor_slices(X_test).batch(32)
# test_pred_scores = get_model_outputs(test_data)
# test_pred_labels = make_prediction(test_pred_scores, threshold, thresholding)


# # # 여기서 부터 수정
# # ## testing on MNIST (Unseen Classes)
# # data_train, data_test = tf.keras.datasets.mnist.load_data()
# # (images_train, labels_train) = data_train
# # (images_test, labels_test) = data_test
# # mnist_test = adjust_images(np.array(images_test))



# # test_batcher = tf.data.Dataset.from_tensor_slices(mnist_test).batch(32)
# # test_scores = get_class_prob(test_batcher)


# # test_mnist_labels = make_prediction(test_scores, threshold, thresholding)

# # ## testing on random noise (Unseen Classes)

# # images = np.random.uniform(0, 1, (10000, 32, 32, 3)).astype(np.float32)
# # test_batcher = tf.data.Dataset.from_tensor_slices(images).batch(32)
# # test_scores = get_class_prob(test_batcher)
# # test_noise_labels = make_prediction(test_scores, threshold, thresholding)

# # test_unseen_labels = np.concatenate([
# #         test_mnist_labels,
# #         test_noise_labels])
    
# # test_pred = np.concatenate([test_pred_labels, test_unseen_labels])
# # test_true = np.concatenate([y_test.flatten(),
# #                             np.ones_like(test_unseen_labels)*m])
    
# # test_macro_f1 = f1_score(test_true, test_pred, average='macro')
# # #print(f1_score(test_true, test_pred, average=None))

# # test_seen_acc = accuracy_score(y_test, test_pred_labels)

# # test_unseen_f1 = np.array([f1_score(np.ones_like(test_unseen_labels), test_unseen_labels == m),
# #                            f1_score(np.ones_like(test_mnist_labels), test_mnist_labels == m),
# #                            f1_score(np.ones_like(test_noise_labels), test_noise_labels == m)])
 
# # print('overall f1: {:.4f}'.format(test_macro_f1))
# # print('seen acc: {:.4f}'.format(test_seen_acc))
# # print('unseen f1: {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}'.format(*test_unseen_f1))

# ##############################################################################################
