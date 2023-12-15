import cv2
import os

def images_to_video(image_path, output_path):
    img_array = []
    imgList = os.listdir(image_path)
    print(imgList)
    imgList.sort(key=lambda x: float(x.split('.')[0]))  
    for count in range(0, len(imgList)): 
        filename = os.path.join(imgList[count])
        img = cv2.imread(image_path + filename)
        if img is None:
            print(filename + " is error!")
            continue
        cv2.putText(img, f"Frame: {count + 1}/{len(imgList)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img_array.append(img)

    height, width, _ = img.shape
    size = (width, height)
    fps = 5  # 设置每帧图像切换的速度
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
def main():
    image_path = "./data/imagenet_attacked_cam/10_1_0.5/"  # 改成你自己图片文件夹的路径,注意一定要有最后的斜杠
    output_path = './data/videos/imagenet_attacked_cam_10_1_0.5.avi'  # 生成视频的保存路径
    images_to_video(image_path, output_path)
 
if __name__ == "__main__":
    main()