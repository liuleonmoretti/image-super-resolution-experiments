import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# 导入你的 GAN 生成器定义
from model.gan import Generator as GANGenerator
import utils


def load_generator(checkpoint_path, scale_factor, cuda_id=0):
    gen = GANGenerator(scale_factor)
    checkpoint = torch.load(checkpoint_path,
                            map_location=f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    if torch.cuda.is_available():
        gen = gen.cuda(cuda_id)
    return gen


def load_lr_image(image_path):
    """直接读取低分（模糊）图，不做下采样"""
    lr = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()  # 归一化到 [0,1]
    lr_t = transform(lr).unsqueeze(0)  # [1, C, H, W]
    return lr_t, lr


def save_sr(output_tensor, save_path):
    out_np = output_tensor.squeeze().cpu().numpy()
    out_np = np.clip(out_np, 0, 1)
    out_img = Image.fromarray((out_np.transpose(1, 2, 0) * 255).astype(np.uint8))
    out_img.save(save_path)


def display_comparison(lr_pil, lr_tensor, sr_tensor):
    # lr_pil: 原始 PIL 低分图
    # lr_tensor: [1,C,Hlr,Wlr], sr_tensor: [1,C,Hsr,Wsr]
    lr_np = np.array(lr_pil) / 255.0
    # 把 lr 插值到 sr 尺寸
    Hsr, Wsr = sr_tensor.shape[2], sr_tensor.shape[3]
    lr_up = lr_pil.resize((Wsr, Hsr), Image.BICUBIC)
    lr_up_np = np.array(lr_up) / 255.0

    sr_np = sr_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    # 拼接并显示
    combined = np.concatenate([lr_up_np, sr_np], axis=1)
    plt.figure(figsize=(8, 4))
    plt.imshow(combined)
    plt.axis('off')
    plt.title('Left: LR upsampled → Right: GAN SR')
    plt.show()


if __name__ == "__main__":
    # 配置
    model_dir = "experiments/gan_model"
    checkpoint_path = os.path.join(model_dir, "best_G.pth.tar")
    params_path = os.path.join(model_dir, "params.json")
    input_path = "imgs/36_114.jpg"
    output_path = "imgs/image_clear_gan.jpg"
    scale_factor = 4
    cuda_id = 0

    # 加载超参
    assert os.path.isfile(params_path), f"No params.json at {params_path}"
    params = utils.Params(params_path)
    params.cuda = torch.cuda.is_available()

    # 加载生成器
    netG = load_generator(checkpoint_path, scale_factor, cuda_id)

    # 读取并预处理低分图
    lr_tensor, lr_pil = load_lr_image(input_path)
    if params.cuda:
        lr_tensor = lr_tensor.cuda(cuda_id)

    # 推理
    with torch.no_grad():
        sr_tensor = netG(lr_tensor)

    # 保存结果
    save_sr(sr_tensor, output_path)
    print(f"Saved super-resolved image to {output_path}")

    # 展示对比
    display_comparison(lr_pil, lr_tensor, sr_tensor)
