from pathlib import Path # 导入Path模块，用于处理文件路径
from PIL import Image # 导入Image模块，用于处理图像
#这是为了统一文件的格式，防止由gif或其他格式图片引起的错误，同时对图片重命名
path = Path.cwd() # 获取当前工作目录的路径
path = path /'pre_data' / 'old' #获取预处理文件目录路径
path = path.iterdir() # 获取'old'目录下的所有文件或子目录的迭代器
for p1 in path: # 遍历'old'目录下的所有文件或子目录
    print(str(p1)) # 打印每个文件或子目录的路径
    i = 0 # 初始化一个计数器为0
    for eachpath in p1.iterdir(): # 遍历每个子目录下的所有文件
        img = Image.open(eachpath) # 打开每个文件为图像对象
        img.save(eachpath, 'PNG') # 以PNG格式保存每个图像对象
        img.close() # 关闭每个图像对象
        eachpath.rename(p1 / f'{i}.png') # 重命名每个文件为子目录名加上计数器值加上'.png'的格式
        i += 1 # 计数器加一