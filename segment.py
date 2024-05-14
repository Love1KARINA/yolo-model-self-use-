from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import os


image_id='001'
#train
model = YOLO('best.pt')

# Train the model
results=model.predict(
        source=f'{image_id}.jpg',
        save=True,  # 保存预测结果
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        conf=0.60,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project='runs/predict',  # 项目名称（可选）
        name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
        save_txt=False,  # 保存结果为 .txt 文件
        save_conf=False,  # 保存结果和置信度分数
        save_crop=False,  # 保存裁剪后的图像和结果
        show_labels=False,  # 在图中显示目标标签
        show_conf=False,  # 在图中显示目标置信度分数
        vid_stride=1,  # 视频帧率步长
        line_width=3,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=True,  # 使用高分辨率的分割掩码
        boxes=False,  # 在分割预测中显示边界框
    )

def sort(pts):
    pts = pts.reshape(-1, 2)
    # 基于x坐标进行排序
    sorted_x = pts[np.argsort(pts[:, 0]), :]
    # 最左边的两个点
    leftmost = sorted_x[:2, :]
    # 最右边的两个点
    rightmost = sorted_x[2:, :]
    if leftmost[0, 1] != leftmost[1, 1]:
        # 最左边两个点的y坐标不同时，按y坐标从小到大排序
        leftmost = leftmost[np.argsort(leftmost[:, 1]), :]
    else:
        # 最左边两个点的y坐标相同时，按x坐标从大到小排序
        leftmost = leftmost[np.argsort(leftmost[:, 0])[::-1], :]
    (tl, bl) = leftmost
    if rightmost[0, 1] != rightmost[1, 1]:
        # 最右边两个点的y坐标不同时，按y坐标从小到大排序
        rightmost = rightmost[np.argsort(rightmost[:, 1]), :]           
    else:
        # 最右边两个点的y坐标相同时，按x坐标从大到小排序
        rightmost = rightmost[np.argsort(rightmost[:, 0])[::-1], :]
    (tr, br) = rightmost
    return np.array([tl, tr, br, bl], dtype="float32")

num=1
if(results is not None):
    for mask in results[0].masks:
        # Convert mask to single channel image
        mask_data = mask.cpu().data.numpy().transpose(1, 2, 0)

        # Execute contour detection
        contours, _ = cv2.findContours(mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 多边形逼近
        polygons = []
        corner_points = []  # 用于存储每个轮廓的四个角

        for contour in contours:
            # 计算轮廓的凸包
            hull = cv2.convexHull(contour)

            # 获取凸包的四个角
            if len(hull) >= 4:
                points = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                if len(points) == 4:
                    corner_points.append(points)

        # 创建一个空白图像
        output_image = np.zeros_like(mask_data)
        #print(corner_points)
            

        # 绘制四个角点并连接形成四边形
        for points in corner_points:
            cv2.polylines(output_image, [points], isClosed=True, color=(255), thickness=2)
            cv2.fillPoly(output_image, [points], (255))
        # 可视化结果
        plt.imshow(output_image, cmap='gray')
        plt.colorbar()
        plt.show()
        
        if(corner_points):
            # 定义源点和目标点
            src_pts = np.float32(corner_points[0])
            src_pts=sort(src_pts)
            # print(src_pts)
            # 定义目标图像尺寸，可以根据分割结果的大小进行调整
            output_width = int(cv2.norm(src_pts[0], src_pts[1]))
            output_height = int(cv2.norm(src_pts[0], src_pts[3]))
            dst_pts = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])

            # 计算透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            original_image = cv2.imread(f'{image_id}.jpg')
            # 应用透视变换
            warped_image = cv2.warpPerspective(original_image, perspective_matrix, (output_width, output_height))
            save_dir = f'./segment/res_{image_id}'
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f'./{save_dir}/warped_image{num}.jpg',warped_image)
            # 可视化结果
            plt.imshow(warped_image)
            plt.axis('off')
            plt.show()
            num=num+1
