import tensorflow as tf
import os

# 消除警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_regression():
    """
    自己实现一个线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        # 1. 准备数据，x 特征值[100, 1], y 目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2. 建立线性回归模型,一个特征x，一个权重w，一个偏置b y = xw + b
        # 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
        # 用变量定义才能优化
        # trainable参数：指定这个变量能否跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w", trainable=True)
        bias = tf.Variable(0.0, name="b", trainable=True)

        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        # 3. 建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4. 梯度下降优化损失, learning_rate: 0~1,2,3,5,7,10
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print("随机初始化参数权重为：%f，偏置为：%f" % (weight.eval(), bias.eval()))

        # 建立事件文件
        # file_writer = tf.summary.FileWriter("D:/project/python/DeepLearning/01验证码识别的实现/event/",
        #                                     graph=sess.graph)

        # 循环训练 运行优化
        for i in range(10000):
            sess.run(train_op)
            print("第%d次优化参数权重为：%f，偏置为：%f" % (i + 1, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    my_regression()
