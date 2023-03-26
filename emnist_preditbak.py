# 导入必要的库
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
import tensorflow
# 加载本地模型文件
model = load_model('emnist_letter.h5')

# 导入一张图片，转换为灰度模式，调整大小为28x28像素
img = tensorflow.keras.preprocessing.image.load_img('m.png', color_mode='grayscale', target_size=(28, 28))

# 将图片转换为numpy数组，归一化并增加一个维度
img = tensorflow.keras.preprocessing.image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# 使用模型进行预测，得到一个概率分布
pred = model.predict(img)

# 找到概率最大的类别，并输出对应的标签
class_index = np.argmax(pred)
class_label = chr(class_index + ord('a') - 1)
print('The predicted class is:', class_label)