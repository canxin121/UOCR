# 导入库
from pathlib import Path

import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from mlxtend.data import loadlocal_mnist
from tensorflow.python.keras.callbacks import EarlyStopping

#多选择特定的数据来增量训练
def process_data(x_train, y_train, x_test, y_test, a1, a2, p1, p2):
    # 找出标签为a1和a2的索引
    train_index_a1_a2 = np.where((y_train == a1) | (y_train == a2))[0]
    test_index_a1_a2 = np.where((y_test == a1) | (y_test == a2))[0]

    # 找出标签为其他的索引
    train_index_other = np.where((y_train != a1) & (y_train != a2))[0]
    test_index_other = np.where((y_test != a1) & (y_test != a2))[0]

    # 随机打乱索引
    np.random.shuffle(train_index_a1_a2)
    np.random.shuffle(test_index_a1_a2)
    np.random.shuffle(train_index_other)
    np.random.shuffle(test_index_other)

    # 取百分之p1的标签为a1和a2的数据
    train_num_a1_a2 = int(len(train_index_a1_a2) * p1)
    test_num_a1_a2 = int(len(test_index_a1_a2) * p1)
    x_train_a1_a2 = x_train[train_index_a1_a2[:train_num_a1_a2]]
    y_train_a1_a2 = y_train[train_index_a1_a2[:train_num_a1_a2]]
    x_test_a1_a2 = x_test[test_index_a1_a2[:test_num_a1_a2]]
    y_test_a1_a2 = y_test[test_index_a1_a2[:test_num_a1_a2]]

    # 取百分之p2的标签为其他的数据
    train_num_other = int(len(train_index_other) * p2)
    test_num_other = int(len(test_index_other) * p2)
    x_train_other = x_train[train_index_other[:train_num_other]]
    y_train_other = y_train[train_index_other[:train_num_other]]
    x_test_other = x_test[test_index_other[:test_num_other]]
    y_test_other = y_test[test_index_other[:test_num_other]]

    # 合并数据
    x_train_new = np.concatenate((x_train_a1_a2, x_train_other), axis=0)
    y_train_new = np.concatenate((y_train_a1_a2, y_train_other), axis=0)
    x_test_new = np.concatenate((x_test_a1_a2, x_test_other), axis=0)
    y_test_new = np.concatenate((y_test_a1_a2, y_test_other), axis=0)

    # 再次打乱数据
    train_perm = np.random.permutation(len(x_train_new))
    test_perm = np.random.permutation(len(x_test_new))
    x_train_new = x_train_new[train_perm]
    y_train_new = y_train_new[train_perm]
    x_test_new = x_test_new[test_perm]
    y_test_new = y_test_new[test_perm]

    return x_train_new, y_train_new, x_test_new, y_test_new


dataroot = Path.cwd() / 'data/emnist/'
train_images, train_labels = loadlocal_mnist(images_path=str(dataroot / 'emnist-letters-train-images-idx3-ubyte'),
                                             labels_path=str(dataroot / 'emnist-letters-train-labels-idx1-ubyte'))
test_images, test_labels = loadlocal_mnist(images_path=str(dataroot / 'emnist-letters-test-images-idx3-ubyte'),
                                           labels_path=str(dataroot / 'emnist-letters-test-labels-idx1-ubyte'))
# 将加载的本地数据集 转换成四维向量，并转换格式为float32并进行归一化

# 将二维的数据转化成思维，前三个维度表示图像的高度、宽度和通道数（例如RGB图像有3个通道），最后一个维度表示样本数量
x_train4D = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
x_test4D = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# 归一化
x_train = x_train4D / 255  # 标准化
x_test = x_test4D / 255

# 对数据的标签进行单独热处理
y_train = np_utils.to_categorical(train_labels)
y_test = np_utils.to_categorical(test_labels)

x_train_new, y_train_new, x_test_new, y_test_new = process_data(x_train,y_train,x_test,y_test,2,5,0.9,0.01)

# 加载模型
model = tf.keras.models.load_model('lettersmore2.h5', compile=False)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证损失
    patience=3,  # 容忍多少个轮次没有改善
    restore_best_weights=True)  # 恢复最佳模型参数
# 训练模型，指定批量大小，迭代次数，验证数据集和回调函数

history = model.fit(x_train, y_train, batch_size=16,  # 使用数据生成器对训练数据进行数据增强
                    epochs=50,  # 迭代次数
                    validation_data=(x_test, y_test),  # 验证数据集
                    callbacks=[early_stopping])

# 保存模型
model.save('lettersmore.h5')
