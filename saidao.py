import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 修改字体
"""
    利用python+opencv 先利用hsv创建黄色赛道区域掩码 然后标记轮廓 并用红线框出实现可视化 实现该效果
    
    参数:
    image_path: 输入图像路径
    
    输出:
    original_image: 原始图像
    marked_image: 用红色轮廓标记赛道区域的图像
    """

#寻廓函数
def extract_track_area(image_path):
    # 读取图像
    original_image = cv2.imread(image_path)
    # 转换颜色空间 BGR to RGB (用于显示) 和 BGR to HSV (用于处理)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)    
    # 定义黄色的HSV范围
    lower_yellow = np.array([14, 100, 100])
    upper_yellow = np.array([36, 255, 255])    
    # 创建黄色区域的掩码
    mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    # 创建标记图像
    marked_image = image_rgb.copy()
    # 如果找到轮廓，筛选并绘制赛道轮廓
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            cv2.drawContours(marked_image, [contour], -1, (255, 0, 0), 3)                
        return image_rgb, marked_image
    else:
        print("未找到黄色赛道区域")
        return 0
    

#可视化
def visualize_results(original, marked):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(original)
    plt.title('原始图像')
    plt.axis('off')  
    plt.subplot(122)
    plt.imshow(marked)
    plt.title('赛道区域检测（红色框标记）')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 输入图像路径
    image_path = r"C:\Users\21155\Desktop\saidao.jpeg"  
    # 提取赛道区域并用红色轮廓标记
    original, marked = extract_track_area(image_path)        
    # 可视化结果
    visualize_results(original, marked)
    # 保存结果
    marked_bgr = cv2.cvtColor(marked, cv2.COLOR_RGB2BGR)
    cv2.imwrite("track_detected.png", marked_bgr)