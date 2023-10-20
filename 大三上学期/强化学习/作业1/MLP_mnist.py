from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = pickle.load(open("mnist.pkl", 'rb'))

# 可视化数据
# x_train_0 = x_train[0]
# y_train_0 = y_train[0]
# plt.figure(figsize=(2,2))
# #cmap=plt.cm.binary 设置为灰度图像
# plt.imshow(x_train_0, cmap=plt.cm.binary)
# plt.suptitle(y_train_0)
# plt.show()

# print(x_train.shape) (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], -1)
# print(x_train.shape) (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)
x_train, y_train = x_train[:6000], y_train[:6000]

#定义分类器
clf = MLPClassifier(activation='logistic', solver='adam', learning_rate_init=0.001, 
                    max_iter=1300, batch_size=800, hidden_layer_sizes=(100,))#len(tuple) = 隐含层大小
#训练
clf.fit(x_train,y_train)
#训练结果
train_score = clf.score(x_train,y_train)
test_score = clf.score(x_test,y_test)

print('MLPClassifier(activation=\'logistic\', batch_size=800, max_iter=1300)' + '\n' + '训练集准确率为%.4f'%train_score)
print('MLPClassifier(activation=\'logistic\', batch_size=800, max_iter=1300)' + '\n' + '测试集准确率为%.4f'%test_score)