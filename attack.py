import argparse
import torch
import cv2
import json
import torchvision
import random
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from utils import GradCAM, show_cam_on_image
import os
from vit_model import vit_base_patch16_224
import shutil

with open("./data/imagenet_class_index.json") as f:
    imagenet_classes = {int(i):x[1] for i, x in json.load(f).items()}

def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        use_cuda = True
        map_location = "cuda"
    else:
        use_cuda = False
        map_location = "cpu"
    
    if model_name == "VGG16" :
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        reshape_transform = None
    elif model_name == 'ResNet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        reshape_transform = None
    elif model_name == "ViT":
        model = vit_base_patch16_224()
        # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
        weights_path = "./data/vit_base_patch16_224.pth"
        model.load_state_dict(torch.load(weights_path, map_location = map_location))
        reshape_transform = ReshapeTransform(model)
    else:
        raise Exception(
            "Please enter a correct model name ")
    return model, use_cuda, reshape_transform

# ViT 计算grad-CAM要用到的函数，其他模型不需要
class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                    self.h,
                                    self.w,
                                    x.size(2))

        # Bring the channels to the first dimension,
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result

# 计算预测类别对于输入的梯度以及grad-cam图
class GradCAMWrapper:
    def __init__(self, model_name = 'VGG16', save_name = None):
        self.model_name = model_name
        self.save_name = save_name
        self.model, self.use_cuda, self.reshape_transform = load_model(self.model_name)
        if self.model_name == 'ViT':
          self.target_layers = [self.model.blocks[-1].norm1]
        elif self.model_name == 'ResNet':
            self.target_layers = [self.model.layer4[2].conv3]
        else:
          self.target_layers = [self.model.features[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers,reshape_transform=self.reshape_transform, use_cuda=self.use_cuda)
        print(f"self.use_cuda = {self.use_cuda}")
    
    def __call__(self, input_tensor, target_category=None):
        predicted_classes, grayscale_cam, grad_of_input = self.cam(input_tensor=apply_normalization(input_tensor), target_category=target_category)
        return predicted_classes, grayscale_cam, grad_of_input
    
    def show_cam(self, img, grayscale_cam, predicted_classes):
        visualization = show_cam_on_image(img,
                                grayscale_cam,
                                use_rgb=True)
        show_images(visualization, predicted_classes, output_path=f'./data/output_{self.model_name}/cam', save_name = self.save_name)
    
    def __normalize(self, image):
        norm = (image - image.mean())/image.std()
        norm = norm * 0.1
        norm = norm + 0.5
        norm = norm.clip(0, 1)
        return norm
    
    def show_grad(self, grad_of_input, predicted_classes):
        grad = self.__normalize(grad_of_input)
        show_images(grad, predicted_classes, output_path=f'./data/output_{self.model_name}/grad', save_name = self.save_name)

 
class ImageNetPredictor:
    def __init__(self, model, data_root='./imagenet', load_images_num = 36, image_size=224):
        self.model = model
        self.data_root = data_root
        self.load_images_num = load_images_num
        self.image_size = image_size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()])
        self.testset = torchvision.datasets.ImageFolder(data_root + '/val', self.transform)

    # 获取模型的预测
    def get_preds(self, X): # X: [b, 224, 224, 3]
        max_value_and_idx =  self.model(apply_normalization(X)).max(dim=1) ### 注意送入模型前执行标准的normalize流程
        return max_value_and_idx[1], max_value_and_idx[0] # 获得预测的label和对应概率
    
    # 保存预测正确的图片和标签
    def save_correct_preds(self, batch_file):
        images = torch.zeros(self.load_images_num, 3, self.image_size, self.image_size) # [100, 3, 224, 224]
        labels = torch.zeros(self.load_images_num).long() # [100,][0,0,...,0]
        preds = labels + 1 # [100, ][1,1,...,1]
        while preds.ne(labels).sum() > 0: # 没全预测对则继续循环 ne:不相等返回1
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)] # 过滤没预测对的， .long-> torch.int64
            for i in list(idx):
                images[i], labels[i] = self.testset[random.randint(0, len(self.testset) - 1)] # 0~49999
            preds[idx], _ = self.get_preds(images[idx])
        torch.save({'images': images, 'labels': labels}, batch_file)

# 载入图片的函数，图片放在一个文件夹下, load_images_num为载入图片数量
def load_images(model, image_folder, load_images_num = 36):

    if image_folder == 'imagenet':
        #从本地加载图片和标签, 硬编码，后面考虑改掉
        batch_file = f'./data/images_labels_{load_images_num}.pth'
        if not os.path.exists(batch_file):
            predictor = ImageNetPredictor(model, load_images_num = load_images_num)
            predictor.save_correct_preds(batch_file = batch_file)
        images_labels = torch.load(batch_file)
        input_tensor = images_labels['images']
        labels = images_labels['labels']
        img_np = np.transpose(input_tensor.numpy(), (0, 2, 3, 1))

    else:
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
        data_transform = transforms.Compose([ transforms.ToTensor()])
        batch_tensors = []
        img_list = []
        for img_path in image_files:
            assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = cv2.resize(np.array(img, dtype=np.uint8), (224, 224))
            img_tensor = data_transform(img)
            img_list.append(img)
            batch_tensors.append(img_tensor)
        input_tensor = torch.stack(batch_tensors, dim=0)  # [batch,3,224,224]
        img_np = np.array(img_list)
        img_np = img_np.astype(dtype=np.float32) / 255
        labels = None
    print(f'input_tensor.shape = {input_tensor.shape}')
    print(f'img_np.shape = {img_np.shape}')
    return input_tensor, img_np, labels
    
def apply_normalization(imgs): # imgs: [h, w, 3] 或 [b, h, w, 3]
    """
    ImageNet图片喂入模型前的标准化处理
    注意在显示图片的时候无需进行这一步，只有作为模型输入的时候需要
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    imgs_tensor = imgs.clone()
    if imgs.dim() == 3:
        for i in range(imgs_tensor.size(0)):
            imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(imgs_tensor.size(1)):
            imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor

def show_images(imgs, titles = None, output_path = None, save_name = None, scale=1.5): 
    '''
    imgs: (batch,224,224,3) numpy array, or [batch,3,224,224] tensor
    titles: list with lenth = batch
    '''
    batch_size = imgs.shape[0]
    num_rows = int(np.ceil(np.sqrt(batch_size)))
    num_cols = int(np.ceil(batch_size / num_rows))
    figsize = (num_cols * scale, (num_rows) * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, image) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(image):# tensor
            image = image.numpy().transpose(1, 2, 0)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
        else:
            image = np.clip(image, 0, 255)
        ax.imshow(image)
        ax.axis("off")  
        if titles:
            ax.set_title(titles[i])
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

class Attack:
    def __init__(self, gradcam, input_tensor, num = 200, ratio = 0.5):
        '''
        gradcam: 一个GradCAMWrapper对象，用于计算每一步的梯度和grad-cam图
        input_tensor: 用于模型输入的tensor, 没有normalize, tensor[batch,3,224,224]
        num: 攻击的像素个数
        ratio: 攻击的强度，即添加的噪声的标准差
        '''
        self.gradcam = gradcam
        self.input_tensor = input_tensor
        self.num = num
        self.ratio = ratio

    def compute_top_indics(self):
        '''
        计算输入数组的前n大的值和位置
        return: 
            top_array: [batch, 3, 224,224]，前n大的值为1，其余为0
            coordinates: 前n大值的位置, shape: (batch, top_num, dim - 1)
        '''
        batch = self.feature_array.shape[0]
        dim = np.ndim(self.feature_array)
        top_array = np.zeros_like(self.feature_array, dtype=int)
        coordinates = np.zeros((batch, self.num, dim - 1), dtype=int)
        for i in range(batch):
            if dim == 3:
                flattened_image = self.feature_array[i].flatten()
                top_indices = np.argpartition(flattened_image, -self.num)[-self.num:]
                coordinates_tmp = np.column_stack(np.unravel_index(top_indices, self.feature_array.shape[1:]))
                coordinates[i] = coordinates_tmp
                top_array[i].flat[top_indices] = 1 
            elif dim == 4:
                flattened_image = self.feature_array[i].reshape(-1, 3)
                top_indices = np.argpartition(flattened_image, -self.num, axis=None)[-self.num:]
                coordinates_tmp = np.column_stack(np.unravel_index(top_indices, self.feature_array.shape[1:]))
                coordinates[i] = coordinates_tmp
                top_array[i].flat[top_indices] = 1  
            else:
                print('不支持的维度')
        if dim == 4:
            top_array = np.transpose(top_array, (0,3,1,2))
        return top_array, coordinates
    
    def add_grey_to_channel(self, top_array):
        """
        将灰度图像的像素值加到输入张量的随机指定通道上。

        参数:
        - top_array: 形状为 (batch_size, height, width) ,每个位置为0或1
        - input_tensor: 形状为 (batch_size, num_channels, height, width) 的输入张量。

        返回:
        形状为 (batch_size, num_channels, height, width) 的新张量。
        """
        num_channels = self.input_tensor.shape[1]
        batch_size = self.input_tensor.shape[0]
        channel_idx = torch.randint(0, num_channels, (batch_size,))
        attacked_tensor = self.input_tensor.clone()
        for i in range(self.input_tensor.shape[0]):
            noise = np.random.normal(loc=0.0, scale= self.ratio, size = top_array[i, :, :].shape)
            attacked_tensor[i, channel_idx[i], :, :] = self.input_tensor[i, channel_idx[i], :, :] + top_array[i, :, :] * noise
        assert attacked_tensor.shape == self.input_tensor.shape
        return attacked_tensor
    
    def get_attacked_input_tensor(self):
        '''攻击后的图片，可直接作为模型输入'''
        noise = np.random.normal(loc=0.0, scale= self.ratio, size = self.input_tensor.shape)
        if self.mask == 'cam' or self.mask == 'grad':
            dim = np.ndim(self.feature_array)
            top_array, _ = self.compute_top_indics()
            if dim == 3:
                attacked_tensor = self.add_grey_to_channel(top_array).float()
            elif dim == 4:
                attacked_tensor = (self.input_tensor + top_array * noise).float()
            else:
                print('不支持的维度')

        else: # self.mode == 'random':
            shape = self.input_tensor.shape
            mask = np.zeros(shape)
            one_indices = np.random.choice(np.prod(shape), size=self.num, replace=False)
            indices = np.unravel_index(one_indices, shape)
            mask[indices] = 1
            attacked_tensor = (self.input_tensor + mask * noise).float()
        assert attacked_tensor.shape == self.input_tensor.shape  
        return attacked_tensor
    
    def predict_and_attack(self, ratio_of_succeed = 0.5, mask = 'cam', max_loop = 1000, output_path = None):
        self.mask = mask
        if os.path.exists(output_path) and os.path.isdir(output_path):
            print(f'删除文件夹：{output_path}')
            shutil.rmtree(output_path)
        original_classes, grayscale_cam, grad_of_input = self.gradcam(self.input_tensor)

        loop_count = 0
        num_differences_list = []
        show_images(self.input_tensor, titles = original_classes, output_path = output_path, save_name = f'{str(loop_count)}.png')
        print(f"攻击方式 = {self.mask}")
        print(f"原始的预测类别 = {original_classes}")
        if self.mask == 'cam':
            self.feature_array = grayscale_cam
        elif self.mask == 'grad':
            self.feature_array = grad_of_input
        elif self.mask == 'random':
            pass
        else:
            raise ValueError("Invalid value for 'mask'. Use 'cam', 'grad', or 'random'.")

        self.attacked_tensor = self.get_attacked_input_tensor()
        num_differences_list.append(0)
        while loop_count <= max_loop:
            loop_count += 1
            print(f'loop_count = {loop_count}')
            predicted_classes, grayscale_cam, grad_of_input = self.gradcam(self.attacked_tensor)
            show_images(self.attacked_tensor, titles = predicted_classes, output_path = output_path, save_name = f'{str(loop_count)}.png')
            num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(predicted_classes, original_classes))
            num_differences_list.append(num_differences)
            if num_differences >= len(predicted_classes) * ratio_of_succeed:
                print(f"攻击之后的预测类别：{predicted_classes}")
                return loop_count, num_differences_list, original_classes, predicted_classes
            else:
                if self.mask == 'cam':
                    self.feature_array = grayscale_cam
                elif self.mask == 'grad':
                    self.feature_array = grad_of_input
                else:
                    pass
                self.input_tensor = self.attacked_tensor
                self.attacked_tensor = self.get_attacked_input_tensor()
        assert loop_count == max_loop + 1, "The loop should have been stopped by now."
        assert len(num_differences_list) == loop_count + 1
        return loop_count, num_differences_list, original_classes, predicted_classes

def parse_args():
    parser = argparse.ArgumentParser(description='Conduct attack experiments')
    parser.add_argument('image_folder', type=str, help='path to image folder')
    parser.add_argument('--model_name', type=str, default='ViT', choices=['ViT', 'VGG16', 'ResNet'], help='model name')
    parser.add_argument('--mask', type=str, default='cam', choices=['grad', 'cam', 'random'], help='mask type')
    parser.add_argument('--ratio', type=float, default=0.5, help='ratio for attack')
    parser.add_argument('--attacked_pixel', type=int, default=200, help='number of attacked pixels')
    parser.add_argument('--ratio_of_succeed', type=float, default=0.5, help='ratio of succeed for attack')
    parser.add_argument('--max_loop', type=int, default=1000, help='max loop for attack')
    parser.add_argument('--output_path', type=str, default=None, help='output path for results')
    return parser.parse_args()

def main():
    args = parse_args()
    image_folder = args.image_folder
    model_name = args.model_name
    mask = args.mask
    ratio = args.ratio
    attacked_pixel = args.attacked_pixel
    ratio_of_succeed = args.ratio_of_succeed
    max_loop = args.max_loop
    output_path = args.output_path if args.output_path else f'./data/{image_folder}_attacked_{mask}/{attacked_pixel}_{ratio}_{ratio_of_succeed}'
    model, use_cuda, reshape_transform = load_model(model_name)
    input_tensor, img, labels = load_images(model, image_folder, load_images_num=36)
    gradcam = GradCAMWrapper(model_name)
    output_path = f'./data/{image_folder}_attacked_{mask}/{ratio}_{ratio_of_succeed}'
    loop_count, num_differences_list, original_classes, predicted_classes = Attack(gradcam, input_tensor, num=attacked_pixel, ratio = ratio).predict_and_attack(ratio_of_succeed = ratio_of_succeed, mask = mask, max_loop=max_loop, output_path = output_path)

if __name__ == '__main__':
    main()
    # 调用方式：python attack.py imagenet --model_name ViT --mask cam --ratio 0.5 --attacked_pixel 200 --ratio_of_succeed 0.5 --max_loop 1000 --output_path ./data/imagenet_attacked_cam/200_0.5_0.5

