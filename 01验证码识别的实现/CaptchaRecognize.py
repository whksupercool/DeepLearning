import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("captcha_dir", "D:/资料/机器学习/数据资料/验证码/captcha.tfrecords",
                           "验证码数据的路径")
tf.app.flags.DEFINE_integer("batch_size", 100, "每批次训练的样本数")


def read_and_decode():
    """
    读取验证码数据API
    :return: image_batch, label_batch
    """
    # 1. 构建文件队列
    file_quene = tf.train.string_input_producer([FLAGS.captcha_dir])

    # 2. 构建阅读器，读取文件内容，默认读取一个样本
    reader = tf.TFRecordReader()

    # 读取内容
    key, value = reader.read(file_quene)

    # terecords格式example，需要解析
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    })

    # 解码内容，字符串内容
    # 先解析图片的特征值
    image = tf.decode_raw(features["image"], tf.uint8)
    # 在解码图片的目标值
    label = tf.decode_raw(features["label"], tf.uint8)
    # print(image, label)

    # 改变形状
    image_reshape = tf.reshape(image, [20, 80, 3])
    label_reshape = tf.reshape(label, [4])
    # print(image_reshape, label_reshape)

    # 进行批处理,每批次读取的样本数 100, 也就是每次训练时候的样本
    image_batch, label_btach = tf.train.batch([image_reshape, label_reshape], batch_size=FLAGS.batch_size, num_threads=1, capacity=FLAGS.batch_size)
    # print(image_batch, label_btach)

    return image_batch, label_btach


def captcharec():
    """
    验证码识别程序
    :return: 
    """
    # 1. 读取验证码数据的文件
    image_batch, label_btach = read_and_decode()

    # 2、通过输入图片特征数据，建立模型，得出预测结果
    # 一层，全连接神经网络进行预测
    # matrix [100, 20 * 80 * 3] * [20 * 80 * 3, 4 * 26] + [104] = [100, 4 * 26]
    

if __name__ == '__main__':
    captcharec()
