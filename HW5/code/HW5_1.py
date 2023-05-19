import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入训练数据
mnist = input_data.read_data_sets("D:/STUDY/CV/MNIST_data/", one_hot=True)
# 每批次读入的训练样本数
batch_size = 100
# 计算一共有多少个批次
# print(mnist.train.num_examples)
n_batch = mnist.train.num_examples // batch_size
n_batch = 5000


# 定义初始化权值函数
def weight_variable(shape, layer_name):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        weights = tf.Variable(initial, name='W')
    return weights


# 定义初始化偏置函数
def bias_variable(shape, layer_name):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        biases = tf.Variable(initial, name='b')
    return biases


# 卷积层
def conv2d(x, W, layer_name):  # 卷积层函数
    outputs = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return outputs


# 池化层
def max_pool_2x2(x, layer_name):  # 池化层函数
    outputs = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# 输入训练数据
with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 28*28
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # keep_prob是保留概率，即我们要保留的结果所占比例，
    # 它作为一个placeholder，在run时传入， 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
    x_image = tf.reshape(xs, [-1, 28, 28, 1], name='x_image')  # 图片高度为1


# 构建神经网络
# conv1 layer #
# 初始化第一个卷积层的权值和偏置
# output_size = 28*28*6
with tf.name_scope('conv1'):
    # 5*5的采样窗口，32个卷积核从1个平面抽取特征
    W_conv1 = weight_variable([5, 5, 1, 32], 'conv1')
    # 每一个卷积核一个偏置值
    b_conv1 = bias_variable([32], 'conv1')
    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'conv1') + b_conv1)

# 进行max_pooling 池化层
# output_size = 14*14*6
with tf.name_scope('pool1'):
    # 进行max_pooling 池化层
    h_pool1 = max_pool_2x2(h_conv1, 'pool1')


# conv2 layer #
# 初始化第二个卷积层的权值和偏置
# output_size = 14*14*64
with tf.name_scope('conv2'):
    # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    W_conv2 = weight_variable([5, 5, 32, 64], 'conv2')
    b_conv2 = bias_variable([64], 'conv2')
    # 把第一个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'conv2') + b_conv2)


# 进行max_pooling 池化层
# output_size = 7*7*64
with tf.name_scope('pool2'):
    # 池化层
    h_pool2 = max_pool_2x2(h_conv2, 'pool2')


# func1 layer #
# 全连接层
# 初始化第一个全连接层的权值
with tf.name_scope('fc1'):
    # func1 layer #
    # 经过池化层后有7*7*64个神经元，全连接层有1024个神经元
    W_fc1 = weight_variable([7 * 7 * 64, 1024], 'fc1')
    b_fc1 = bias_variable([1024], 'fc1')
    # 把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # 求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout防止过拟合, keep_prob用来表示神经元的输出概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer #
# 初始化第二个全连接层
with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10], 'fc2')
    b_fc2 = bias_variable([10], 'fc2')

# 输出层
# 计算输出
with tf.name_scope('output'):
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))


# 使用AdamOptimizer进行优化训练步骤
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 测试准确率
with tf.name_scope('test'):
    # 结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    # 求准确率(tf.cast将布尔值转换为float型)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 创建会话
sess = tf.Session()
# 初始化变量
sess.run(tf.global_variables_initializer())

# 训练并测试数据
for batch in range(n_batch):
    # 分批训练数据
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # 在训练数据是50的倍数时，输出当前训练数据的大小和测试正确率
    if (batch+1) % 50 == 0:
        acc = sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})
        print("训练数据：" + str((batch+1)*batch_size) + "识别正确率：" + str(acc))

sess.close()