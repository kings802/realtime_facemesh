# realtime_facemesh
基于mediapipe、OpenCV的实时人脸网格的python解决方案

MediaPipe Face Landmarker 任务可检测图像和视频中的人脸标志和面部表情。可以使用此任务来识别人类面部表情、应用面部滤镜和效果以及创建虚拟头像。此任务使用可以处理单个图像或连续图像流的机器学习 (ML) 模型。<br>
该任务输出 3 维面部标志、混合形状分数（表示面部表情的系数）以实时推断详细的面部表面，以及转换矩阵以执行效果渲染所需的转换。<br>

配置选项：<br>
![1691486243538](https://github.com/kings802/realtime_facemesh/assets/19601216/5d91b21c-b32e-4a5c-8d32-9dac9164342a)


本文设置running_mode= LIVE_STREAM，实时视频帧<br>
使用Face mesh model模型添加面部的完整映射。该模型输出 478 个 3 维人脸特征点的估计值。<br>

输出效果：<br>
![image](https://github.com/kings802/realtime_facemesh/assets/19601216/4ad5adb7-ee6f-4abb-84f9-2248d804c5f5)
