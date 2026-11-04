import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import model.drrn as Drrn
import model.srcnn as Srcnn
import utils

model_directory = {'srcnn': Srcnn, 'drrn': Drrn}


def load_model_from_checkpoint(model, checkpoint_path, optimizer=None):
    checkpoint = torch.load(checkpoint_path)

    # 加载模型的权重
    model.load_state_dict(checkpoint['state_dict'])

    # 如果有优化器状态，加载它
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, optimizer


def load_and_preprocess_image(image_path, image_size=(144, 144)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return image_tensor


def inference(model, image_tensor, cuda_id=0):
    if torch.cuda.is_available():
        model = model.cuda(cuda_id)
        image_tensor = image_tensor.cuda(cuda_id)

    with torch.no_grad():
        output = model(image_tensor)
        return output


def save_image(output, save_path):
    output = output.squeeze().cpu().numpy()  # 去除batch维度并转回numpy
    output = np.clip(output, 0, 1)  # 将像素值限制在[0, 1]范围
    output_image = Image.fromarray((output * 255).astype(np.uint8).transpose(1, 2, 0))  # 转换为图片格式
    output_image.save(save_path)


def display_image(image_tensor):
    image = image_tensor.squeeze().cpu().numpy()  # 去除batch维度并转回numpy
    image = np.clip(image, 0, 1)
    plt.imshow(image.transpose(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    # 设置路径和超参数
    model_name = "drrn"
    model_path = 'experiments/'+model_name+'_model/best.pth.tar'
    image_path = 'imgs/144_114.jpg'  # 修改为你想要推理的图像路径
    save_path = 'imgs/image_clear_'+model_name+'.jpg'  # 输出图像保存路径

    model_dir = 'experiments/'+model_name+'_model'
    net = model_directory[model_name]

    # Load the parameters from json file
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    cuda_id = 0

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # 加载模型
    model = net.Net(params).cuda(cuda_id) if params.cuda else net.Net(params)
    optimizer = torch.optim.Adam(model.parameters())  # 根据需要调整优化器
    model, optimizer = load_model_from_checkpoint(model, model_path, optimizer)

    # 加载并预处理图像
    image_tensor = load_and_preprocess_image(image_path)

    # 推理
    output = inference(model, image_tensor)

    # save & show SR image
    save_image(output, save_path)

    # — Side-by-side comparison —
    orig = Image.open(image_path).convert('RGB')
    sr = Image.open(save_path).convert('RGB')

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Low-res Input')
    plt.imshow(orig)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Super-resolved Output')
    plt.imshow(sr)
    plt.axis('off')

    plt.show()


