import tkinter as tk
from ttkbootstrap import Style
from tkinter import ttk
from ttkbootstrap.constants import *
import os
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from pathlib import Path
import numpy as np
import tensorflow
import tensorflow as tf
from PIL import ImageDraw, Image

from more import crop_image

digits = []
num_row = 1
# map:0mnist,1letter,2class
map = '0'
# model:0best,emnist_letter
modelchoice = '0'
mappings = {}
root = Path(__file__).parent / r'data\emnist'
modelroot = Path(__file__).parent / r"models"
model = tf.keras.models.load_model(str(modelroot/'best.h5'))
def setmodelchoice(modelchoice):
    global model
    if modelchoice == '0':
        model = tf.keras.models.load_model(str(modelroot/'best.h5'))
    elif modelchoice == '1':
        model = tf.keras.models.load_model(str(modelroot / 'lettersmore3.h5'))
    elif modelchoice == '2':
        model = tf.keras.models.load_model(str(modelroot/'emnist.h5'))



def setmatchoice(map):
    global mappings
    if map == '1':
        # 读取emnist-letters-mapping.txt文件
        with open(str(root / 'emnist-letters-mapping.txt')) as f:
            lines = f.readlines()
        # 创建一个字典，将数字映射到字母
        mappings = {}
        for line in lines:
            index, upper, lower = line.split()  # 分割三个值
            label = chr(int(lower))  # 将大写字母的ASCII码转换为字符
            mappings[int(index)] = label  # 索引从1开始
    elif map == '2':
        with open(str(root / 'emnist-byclass-mapping.txt')) as f:
            lines = f.readlines()
        # 创建一个字典，将数字映射到字符
        mappings = {}
        for line in lines:
            index, label = line.split()  # 分割两个值
            label = chr(int(label))  # 将ASCII码转换为字符
            mappings[int(index)] = label  # 索引从0开始


def predict_digit(img):
    if modelchoice == '0':
        img = img.reshape(1, 28, 28, 1)
        res = model.predict([img])[0]
        return np.argmax(res), max(res)
    else:
        if modelchoice == '1':
            image_data = np.rot90(img, -1)
            img = np.fliplr(image_data)
        img = tensorflow.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        # 使用模型进行预测，得到一个概率分布
        pred = model.predict(img)
        # 找到概率最大的类别，并输出对应的标签

        class_index = np.argmax(pred)
        class_label = mappings[class_index]
        print(f"The predicted letter is {class_label} with probability {max(pred)}.")
        return class_label, max(max(pred))


class App:
    def __init__(self):
        self.pen_color = "black"
        self.root = tk.Tk()
        self.mode = 'draw'
        self.root.withdraw()  # 隐藏根窗口

        # 创建 ttkbootstrap 窗口
        self.style = Style(theme='yeti')
        self.start_window = tk.Toplevel(self.root)
        self.start_window.geometry("300x300")
        self.start_window.title("开始")
        self.start_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 创建 PhotoImage 对象
        bg_image = tk.PhotoImage(file="./envdav/background.png")
        # 创建标签小部件
        bg_label = ttk.Label(self.start_window, image=bg_image)
        # 把图片对象作为标签对象的一个属性
        bg_label.image = bg_image
        # 使用 place 方法放置标签小部件

        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # 创建按钮小部件
        btn_canvas = ttk.Button(self.start_window, text="画布识别", command=self.show_canvas)
        btn_canvas.place(relx=0.5, rely=0.3, anchor='center', width=120, height=50)

        btn_import = ttk.Button(self.start_window, text="导入图片识别", command=self.import_image_recognition)
        btn_import.place(relx=0.5, rely=0.5, anchor='center', width=120, height=50)
        # 创建标签和下拉菜单小部件
        self.label_map = ttk.Label(self.start_window, text="选择map")
        self.label_map.place(relx=0.15, rely=0.7, anchor='center')

        self.map_var = tk.StringVar()  # 创建一个变量，用来保存选择的值
        self.map_var.set("mnist")  # 设置默认值
        self.option_menu = ttk.OptionMenu(self.start_window, self.map_var, 'mnist', 'mnist', 'emnist-letters', 'emnist-byclass')
        self.option_menu.place(relx=0.5, rely=0.7, anchor='center', width=120, height=30)

        # 创建一个输入框，使用 bootstyle 参数设置样式
        self.entry = ttk.Spinbox(self.start_window, from_=1, to=4)
        self.entry.place(relx=0.5, rely=0.9, anchor=CENTER, width=120, height=30)
        self.entry.insert(0,"1")

        # 创建一个标签，显示输入框的描述
        self.label = ttk.Label(self.start_window,text="输入行数1~4")
        self.label.place(relx=0.15, rely=0.90, anchor=tk.CENTER)

        # 创建一个确定按钮，绑定点击事件，使用 bootstyle 参数设置样式
        self.button = ttk.Button(self.start_window, text="确定", command=self.get_num_row)
        self.button.place(relx=0.80, rely=0.9, anchor=CENTER, width=50, height=35)

        if not os.path.exists('recognized'):
            os.mkdir('recognized')

        if not os.path.exists('./envdav/order.txt'):
            with open('./envdav/order.txt', 'w') as f:
                f.write('0')

        with open('./envdav/order.txt', 'r') as f:
            self.order = int(f.read())

    # 创建一个函数，用来根据选择的值设置map变量
    def set_map(self):
        global map, modelchoice  # 声明全局变量
        map = self.map_var.get()  # 获取选择的值
        if map == 'emnist-letters':
            map = '1'
        elif map == 'emnist-byclass':
            map = '2'
        elif map == 'mnist':
            map = '0'
        # 调用更改mapping的函数
        setmatchoice(map)
        modelchoice = map
        setmodelchoice(modelchoice)
        print("map =", map)  # 打印测试

    def get_num_row(self):
        self.set_map()
        global num_row  # 声明全局变量
        try:
            # 尝试将输入框的值转换为整数
            num_row = int(self.entry.get())
            # 判断输入的值是否在合理的范围内
            if num_row > 4 or num_row <= 0:
                # 如果不合理，弹出提示框
                messagebox.showerror("错误", "请输入0到4之间的整数")
            else:
                # 如果合理，弹出提示框
                messagebox.showinfo("成功", "你输入的行数是{}".format(num_row))
        except ValueError:
            # 如果转换失败，说明输入的不是数字，弹出提示框
            messagebox.showerror("错误", "请输入数字")

    def printf(self, string):
        self.printf_window = Toplevel(self.root)
        self.printf_window.title('识别结果')
        self.printf_window.geometry("250x100")
        label = Label(self.printf_window, text=string, anchor=CENTER, justify=CENTER)
        label.pack(expand=YES)

    # noinspection PyAttributeOutsideInit

    def on_closing(self):
        self.root.quit()

    def show_canvas(self):
        self.mode = 'draw'
        # 显示画布和其他按钮
        self.start_window.withdraw()
        self.canvas_window = Toplevel(self.root)
        self.canvas_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.canvas_window.title("画布")

        self.canvas = Canvas(self.canvas_window, bg='white', width=1240, height=200 * num_row)
        self.canvas.pack()
        for i in range(1, num_row):
            self.canvas.create_line(0, 200 * i, 1240, 200 * i, dash=(5, 5), tag='guide')
        self.image1 = Image.new("RGB", (1240, 200 * num_row), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

        button_frame = Frame(self.canvas_window)
        button_frame.pack(side=BOTTOM, anchor=CENTER)

        btn_recognize = Button(button_frame, text="识别", command=self.recognize)
        btn_recognize.pack(side=LEFT)

        btn_clear = Button(button_frame, text="清除", command=self.clear_all)
        btn_clear.pack(side=LEFT)

        btn_back = Button(button_frame, text="返回", command=self.back_to_start)
        btn_back.pack(side=LEFT)

        # 创建一个画笔颜色的变量
        self.pen_color = "black"

        # 创建一个画笔按钮
        btn_pen = Button(button_frame, text="画笔", command=self.use_pen)
        btn_pen.pack(side=RIGHT)

        # 创建一个橡皮按钮
        btn_eraser = Button(button_frame, text="橡皮", command=self.use_eraser)
        btn_eraser.pack(side=RIGHT)

    # 定义画笔按钮的回调函数
    def use_pen(self):
        self.pen_color = "black"

    # 定义橡皮按钮的回调函数
    def use_eraser(self):
        self.pen_color = "white"

    # 在绘图函数中使用画笔颜色的变量
    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, outline=self.pen_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.pen_color, outline=self.pen_color)

    def back_to_start(self):
        # 销毁画布识别窗口
        self.canvas_window.destroy()

        # 显示开始窗口
        self.start_window.deiconify()

    def import_image_recognition(self):
        self.mode = 'import'
        global digits
        imgpath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if imgpath:
            # 打开图片文件并将其转换为 NumPy 数组
            img = Image.open(imgpath)
            img_array = np.array(img)
            self.image1 = img_array
            # 调用 recognize 方法处理图片
            self.recognize()
        digits.clear()

    def recognize(self):
        img = self.image1
        if hasattr(self, 'printf_window'):
            self.printf_window.destroy()
        if self.mode == 'draw':
            self.canvas.delete('guide')
            img = self.image1
            for i in range(1, num_row):
                self.canvas.create_line(0, 200 * i, 1240, 200 * i, dash=(5, 5), tag='guide')
        cropped_images = crop_image(img, num_row)
        global digits
        if len(cropped_images) == 0:
            self.printf('没有识别到数字')
        else:
            for img in cropped_images:
                digit, prediction = predict_digit(img)
                digits.append(digit)
                print(f'预测结果: {digit}, 置信度: {prediction * 100:.2f}%')
                filename = f'recognized/{digit}_{self.order}.png'
                print(f'保存图片到: {filename}')
                pil_img = Image.fromarray(img)
                pil_img.save(filename)

                with open('./envdav/order.txt', 'w') as f:
                    f.write(str(self.order + 1))

                self.order += 1
            strdigits = [str(x) for x in digits]
            strdigits = ''.join(strdigits)
            self.printf(f'识别到的数字是:{strdigits}')
            digits.clear()

    def clear_all(self):
        if hasattr(self, 'printf_window'):
            self.printf_window.destroy()
        global digits
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1240, 200 * num_row], fill='white')
        digits.clear()
        for i in range(1, num_row):
            self.canvas.create_line(0, 200 * i, 1240, 200 * i, dash=(5, 5), tag='guide')


app = App()
app.root.mainloop()
