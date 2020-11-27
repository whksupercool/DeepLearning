import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 调用matplotlib的3D绘图功能

# 设置学习率
learning_rate = 0.01
# 设置最小误差
threshold = 0.01
# 构造训练数据
x1_data = np.random.randn(100).astype(np.float32)
x2_data = np.random.randn(100).astype(np.float32)
y_data = 2 * x1_data + 3 * x2_data + 1
# 构建模型
weight1 = tf.Variable(1.)
weight2 = tf.Variable(1.)
bias = tf.Variable(1.)
x1_ = tf.placeholder(tf.float32)
x2_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
# 构建模型Y = weight1 * X1 + weight2 * X2 + Bias
y_model = tf.add(tf.add(tf.multiply(weight1, x1_), tf.multiply(weight2, x2_)), bias)
# 采用均方差做为损失函数
loss = tf.reduce_mean(tf.pow((y_model - y_), 2))
# 使用随机梯度下降算法
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    # 参数初始化
    sess.run(tf.global_variables_initializer())
    # 开始训练
    print("Start training!")
    flag = 1
    while (flag):
        # 使用zip进行嵌套，表示三个参数
        for (x, y) in zip(zip(x1_data, x2_data), y_data):
            sess.run(train_op, feed_dict={x1_: x[0], x2_: x[1], y_: y})
            # 当训练损失低于threshold时停止训练
            if sess.run(loss, feed_dict={x1_: x[0], x2_: x[1], y_: y}) < threshold:
                flag = 0
    w1 = sess.run(weight1)
    w2 = sess.run(weight2)
    b = sess.run(bias)
    print('n')
    print('线性回归方程为：')
    print("Y = %f * X1 + %f * X2 + %f " % (w1, w2, b))
    print('n')
    # 绘制模型图
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # X, Y = np.meshgrid(x1_data, x2_data)
    # Z = sess.run(weight1) * (X) + sess.run(weight2) * (Y) + sess.run(bias)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    # ax.contourf(X, Y, Z, zdir='z', offset=-1, camp=plt.cm.hot)
    # ax.set_title('analysis')
    # ax.set_ylabel('salary')
    # ax.set_xlabel('age')
    # ax.set_zlabel('amount')
    # ax.set_zlim(-1, 1)
    # plt.show()
