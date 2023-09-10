import torch
import uuid
import cv2
import json
import torch.nn as nn
from datetime import datetime
import torchvision
import requests
import random
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from utils import GradCAM, show_cam_on_image
import os
from torch.utils.data import DataLoader
from vit_model import vit_base_patch16_224

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
        weights_path = "./vit_base_patch16_224.pth"
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
    def __init__(self, model_name = 'VGG16'):
        self.model_name = model_name
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
        show_images(visualization, predicted_classes, output_path=f'./data/output_{self.model_name}/cam')
    
    def __normalize(self, image):
        norm = (image - image.mean())/image.std()
        norm = norm * 0.1
        norm = norm + 0.5
        norm = norm.clip(0, 1)
        return norm
    
    def show_grad(self, grad_of_input, predicted_classes):
        grad = self.__normalize(grad_of_input)
        show_images(grad, predicted_classes, output_path=f'./data/output_{self.model_name}/grad')


# 获取模型预测正确的100张图片   
class ImageNetPredictor:
    def __init__(self, model, data_root='./imagenet', image_size=224):
        self.model = model
        self.data_root = data_root
        self.image_size = image_size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()])
        self.testset = torchvision.datasets.ImageFolder(data_root + '/val', self.transform)

    def get_preds(self, X): # X: [b, 224, 224, 3]
        """获取model的预测""" 
        max_value_and_idx =  self.model(apply_normalization(X)).max(dim=1) ### 注意送入模型前执行标准的normalize流程
        return max_value_and_idx[1], max_value_and_idx[0] # 获得预测的label和对应概率

    def save_correct_preds(self, num_runs=36, batch_file='./data/images_labels.pth'):
        images = torch.zeros(num_runs, 3, self.image_size, self.image_size) # [100, 3, 224, 224]
        labels = torch.zeros(num_runs).long() # [100,][0,0,...,0]
        preds = labels + 1 # [100, ][1,1,...,1]
        while preds.ne(labels).sum() > 0: # 没全预测对则继续循环 ne:不相等返回1
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)] # 过滤没预测对的， .long-> torch.int64
            for i in list(idx):
                images[i], labels[i] = self.testset[random.randint(0, len(self.testset) - 1)] # 0~49999
            preds[idx], _ = self.get_preds(images[idx])
        torch.save({'images': images, 'labels': labels}, batch_file)

# 载入图片的函数，图片放在一个文件夹下
def load_images(model, image_folder = None):

    if image_folder == 'imagenet':
        #从本地加载图片和标签
        batch_file = './data/images_labels.pth'
        if not os.path.exists(batch_file):
            predictor = ImageNetPredictor(model)
            predictor.save_correct_preds()
        images_labels = torch.load(batch_file)
        input_tensor = images_labels['images']
        labels = images_labels['labels']
        img_np = np.transpose(input_tensor.numpy(), (0, 2, 3, 1))

    else:
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
        data_transform = transforms.Compose([ 
                                            transforms.ToTensor()])
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
    print(f'input_tensor,shape = {input_tensor.shape}')
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

def show_images(imgs, titles=None, output_path=None, scale=1.5): 
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
            ax.imshow(image.numpy().transpose(1, 2, 0))
        else:
            ax.imshow(image)
        ax.axis("off")  
        if titles:
            ax.set_title(titles[i])
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{ str(uuid.uuid4())}.png"
        output_path = os.path.join(output_path, filename)
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == '__main__':
    # image_folder = './data/images'
    image_folder = 'imagenet'
    model_names = ['VGG16', 'ResNet', 'ViT']
    for i, model_name in enumerate(model_names):
        model, use_cuda, reshape_transform = load_model(model_name)
        input_tensor, img, labels = load_images(model, image_folder)
        if i == 0:
            show_images(input_tensor, titles=[imagenet_classes[int(label)] for label in labels], output_path='./data/input')
        gradcam = GradCAMWrapper(model_name)
        predicted_classes, grayscale_cam, grad_of_input = gradcam(input_tensor)
        # print(predicted_classes)
        gradcam.show_cam(img, grayscale_cam, predicted_classes)
        gradcam.show_grad(grad_of_input, predicted_classes)