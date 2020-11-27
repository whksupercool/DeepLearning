# 1:导入所需要的软件包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# tf.disable_eager_execution()

'''

使用Tensorflow实现多元线性回归

'''


# 2：因为各特征的数据范围不同，需要归一化特征数据。为此定义一个归一化函数
# 另外，这里添加一个额外的固定输入值将权重和偏置结合起来。
# 为此定义函数append_bias_reshape()。该技巧可简化编程
def normalize(X):
    '''
    归一化数组 X
    np.mean:计算均值
    np.std：计算标准差
    '''
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


def append_bias_reshape(features, labels):
    '''
    param features: 对于矩阵来说，shape[0]:表示矩阵的行数
                                shape[1]:表示矩阵的列数
    '''
    m = features.shape[0]
    n = features.shape[1]
    '''
    np.c_:按行将矩阵组合起来
    '''
    x = np.reshape(np.c_[np.ones(m), features], [m, n + 1])
    y = np.reshape(labels, [m, 1])
    return x, y


# 3：加载波士顿房价数据集，并划分为X_train,Y_train
# 可以选择这里对数据进行归一化处理，也可以添加偏置并对网络数据重构

boston = load_boston()
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
X_train, Y_train = append_bias_reshape(X_train, Y_train)
# 训练示例数
m = len(X_train)
# 特征+偏置的数量
n = 13 + 1
# 4：为训练数据声明Tensorflow占位符，观测占位符X的形状变化
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y')

# 5：为权重和偏置创建Tensorflow变量，通过随机数初始化权重
w = tf.Variable(tf.random_normal([n, 1]))
b = tf.Variable(tf.zeros(1))

# 6:定义用于预测的线性回归模型。需要矩阵乘法完成任务

Y_hat = tf.matmul(X, w)

# 7:为了更好的求微分，定义损失函数
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))

# 8:选择正确的优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 9：定义初始化操作符
init_op = tf.global_variables_initializer()
total = []

# 10:开始计算图

with tf.Session() as sess:
    sess.run(init_op)
    # writer = tf.summary.FileWriter('graphs2', sess.graph)
    for i in range(100):
        l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
        total.append(l)
        print('Epoch {0}:Loss {1}'.format(i, l))
        # writer.close()
        w_value, b_value = sess.run([w, b])

# 11:绘制损失函数
plt.plot(total)
plt.show()

# 12：从模型中学到的系数来预测房价
N = 500
X_new = X_train[N, :]
Y_pred = (np.matmul(X_new, w_value) + b_value).round(1)
print('Predicted value:${0} Actual value: / ${1}'.format(Y_pred[0] * 1000, Y_train[N] * 1000, '\nDone'))
