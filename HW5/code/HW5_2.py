import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import time
import cifar10_input
import math


# 每批次读入的训练样本数
batch_size = 100
# 最大迭代轮数
max_steps = 5000

# 数据所在路径
data_dir = './cifar10_data/cifar-10-batches-bin'
# 读入图像数据及标签，并进行数据增强
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# 裁剪图片正中间的24*24大小的区块并进行数据标准化操作
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 下载数据库
# cifar10.maybe_download_and_extract()

## 初始化 weight 函数
def weight_variable(shape, stddev, wd):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=stddev)
        weights = tf.Variable(initial, name='W')
    # var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return weights


# 定义初始化偏置函数
def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        biases = tf.Variable(initial, name='b')
    return biases


# 卷积层
def conv2d(x, W):  # 卷积层函数
    outputs = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return outputs


# 池化层
def max_pool_2x2(x):  # 池化层函数
    outputs = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    return outputs


# 输入层
with tf.name_scope('inputs'):
    # 定义placeholder
    # 注意此处输入尺寸的第一个值应该是batch_size而不是None
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])


# 构建神经网络
# conv1 layer #
# 初始化第一个卷积层的权值和偏置
with tf.name_scope('conv1'):
    weight1 = weight_variable(shape=[5, 5, 3, 64],stddev=5e-2,wd=0.0)
    bias1 = bias_variable([64])
    kernel1 = conv2d(image_holder, weight1)
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))

# 进行max_pooling 池化层
with tf.name_scope('pool1'):
    pool1 = max_pool_2x2(conv1)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# conv2 layer #
# 初始化第二个卷积层的权值和偏置
with tf.name_scope('conv2'):
    weight2 = weight_variable([5, 5, 64, 64], stddev=5e-2, wd=0.0)
    bias2 = bias_variable([64])
    kernel2 = conv2d(norm1, weight2)
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))

# 进行max_pooling 池化层
with tf.name_scope('pool2'):
    pool2 = max_pool_2x2(conv2)
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# func1 layer #
# 全连接层
# 初始化第一个全连接层的权值
with tf.name_scope('fc1'):
    reshape = tf.reshape(pool2, [batch_size, -1])  # 将每个样本reshape为一维向量
    dim = reshape.get_shape()[1].value  # 取每个样本的长度
    weight3 = weight_variable([dim, 384], stddev=0.04, wd=0.004)
    # bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    bias3 = bias_variable([384])
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)


# func2 layer #
# 初始化第二个全连接层
with tf.name_scope('fc2'):
    weight4 = weight_variable([384, 192], stddev=0.04, wd=0.004)
    # bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    bias4 = bias_variable([192])
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# func2 layer #
# 初始化第三个全连接层
with tf.name_scope('fc3'):
    weight5 = weight_variable([192, 10],stddev=1/192.0, wd=0.0)
    # bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    bias5 = bias_variable([10])
    logits = tf.add(tf.matmul(local4, weight5), bias5)

# 定义损失函数loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# 定义loss
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) # 定义优化器
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)


# 定义会话并开始迭代训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动图片数据增强的线程队列
tf.train.start_queue_runners()


# 训练并测试数据
for batch in range(max_steps):
    # 分批训练数据
    image_batch, label_batch = sess.run([images_train, labels_train]) # 获取训练数据
    sess.run([train_op, loss],feed_dict={image_holder: image_batch,label_holder: label_batch})
    # 在训练数据是50的倍数时，输出当前训练数据的大小和测试正确率
    if batch % 50 == 0:
        # 在测试集上测评准确率
        num_examples = 1000  # 测试用数据量
        num_iter = int(math.ceil(num_examples / batch_size))
        true_count = 0  # 计算正确数量
        total_sample_count = num_iter * batch_size  # 一共测试的数据
        step = 0
        while step < num_iter:
            image_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
            true_count += np.sum(predictions)
            step += 1
        precision = true_count / total_sample_count # 正确率
        print("训练数据集：" + str(batch*batch_size) + "正确率：%.3f" % precision)