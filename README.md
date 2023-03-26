# UOCR
UOCR: User-friendly Optical Character Recognition Program

已经实现；  
1.手写数字和字母数据集建立和模型训练
2.ui界面，提供模型选项，提供画板多行识别和导入图片识别

问题：  
1.byclass模型精度不高（模型可以自己练）  

todo：  
1.导入图片自动识别行数
2.导出识别内容保持原图片格式
本项目没有什么实际作用，主要用于学习python和深度学习方面的知识。  
本项目包括  
1.mnist数据集的创建：首先将各个数字的照片分好类放在old中，通过prepic.py可以将其处理成可以用于转化为ubyte类型数据集的黑底白字28*28图  
进而分配为训练集train和测试集test两部分，然后通过pictoubyte.py转化为mnist同格式的训练集和测试集。  
2.recognize.py可以提供一个tk窗口来进行使用，注意先输入行数（默认1）模型(默认mnist)，此py依赖more.py进行数字分割。   
3.另有nonebot2插件接入cqhttp，马上发布（这有什么用？）。  

![image](https://github.com/canxin121/UOCR/blob/main/envdav/show%20(1).png)  
![image](https://github.com/canxin121/UOCR/blob/main/envdav/show%20(2).png)  
![image](https://github.com/canxin121/UOCR/blob/main/envdav/show%20(3).png)  
![image](https://github.com/canxin121/UOCR/blob/main/envdav/show%20(4).png)  
![image](https://github.com/canxin121/UOCR/blob/main/envdav/show%20(5).png)  
