"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.data_loader as data_loader
from evaluate_gan import evaluate
import torchvision.transforms.functional as F
from skimage import io
from logger import Logger
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import model.srcnn as Srcnn
import model.drrn as Drrn
import model.gan as GAN

model_directory = {'srcnn': Srcnn, 'drrn': Drrn, 'gan': GAN }
# torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/4_3_test', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model', default=None)
parser.add_argument('--cuda', default=None)
parser.add_argument('--optim', default='adam')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

opt = parser.parse_args()
net = model_directory[opt.model]
# if net == GAN:
#     from net import Generator, Discriminator
from loss import GeneratorLoss, GeneratorLoss_adv, GeneratorLoss_ssim, GeneratorLoss_notv

global_loss_g = []
global_loss_d = []


def train(netG, netD, optimG, optimD, loss_fn, dataloader, metrics, params, cuda_id):
    netG.train()
    netD.train()
    g_loss_avg = utils.RunningAverage()
    d_loss_avg = utils.RunningAverage()
    mse_loss_avg = utils.RunningAverage()
    summ = []

    with tqdm(total=len(dataloader)) as t:
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            if params.cuda:
                lr_imgs, hr_imgs = lr_imgs.cuda(cuda_id), hr_imgs.cuda(cuda_id)
            lr_imgs, hr_imgs = Variable(lr_imgs), Variable(hr_imgs)

            # 1) D 更新
            # 只 forward 一次 G
            fake_imgs = netG(lr_imgs)
            fake_detached = fake_imgs.detach()
            netD.zero_grad()
            real_out = netD(hr_imgs).mean()
            fake_out = netD(fake_detached).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward()
            optimD.step()

            # 2) G 更新
            netG.zero_grad()
            # 直接复用上面 forward 得到的 fake_imgs
            fake_out_for_g = netD(fake_imgs).mean()
            g_loss = loss_fn(fake_out_for_g, fake_imgs, hr_imgs)
            g_loss.backward()
            optimG.step()

            # 计算 MSE, PSNR, SSIM
            mse_loss = ((hr_imgs * 255 - fake_imgs * 255) ** 2).mean()
            # 注意 metrics 函数顺序：第一个参数是网络输出，第二个是 ground truth
            if i % params.save_summary_steps == 0:
                out_np = fake_imgs.detach().cpu().numpy()
                hr_np = hr_imgs.detach().cpu().numpy()
                summary = {m: metrics[m](out_np, hr_np) for m in metrics}
                summary.update({'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'mse_loss': mse_loss.item()})
                summ.append(summary)

            # 更新进度条
            g_loss_avg.update(g_loss.item())
            d_loss_avg.update(d_loss.item())
            mse_loss_avg.update(mse_loss.item())
            t.set_postfix(g='{:05.3f}'.format(g_loss_avg()),
                          d='{:05.3f}'.format(d_loss_avg()),
                          mse='{:05.3f}'.format(mse_loss_avg()))
            t.update()

    # 打印本 epoch 指标
    metrics_mean = {k: np.mean([x[k] for x in summ]) for k in summ[0]}
    logging.info("- Train metrics: " + "; ".join(f"{k}:{v:05.3f}" for k, v in metrics_mean.items()))


def train_and_evaluate(netG, netD, train_dataloader, val_dataloader, optimG, optimD, loss_fn, metrics, params,
                       model_dir,
                       restore_file=None, cuda_id=0):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path_g = os.path.join(args.model_dir, 'best_g' + '.pth.tar')
        restore_path_d = os.path.join(args.model_dir, 'best_d' + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path_g))
        utils.load_checkpoint(restore_path_g, netG, optimG)
        utils.load_checkpoint(restore_path_d, netD, optimD)

    best_val_acc = 0.0
    # train add logger,epoch two parameters
    #     logger = Logger('./logs')

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(netG, netD, optimG, optimD, loss_fn, train_dataloader, metrics, params, cuda_id)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(netG, netD, loss_fn, val_dataloader, metrics, params, cuda_id)
        # print ('after val --------')

        val_acc = val_metrics['PSNR']
        is_best = val_acc >= best_val_acc

        # Save weights
        # save G
        flag = 'G'
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': netG.state_dict(),
                               'optim_dict': optimG.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir, flag=flag)
        flag = 'D'
        # save D
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': netD.state_dict(),
                               'optim_dict': optimD.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir, flag=flag)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        if epoch % 100 == 0 and epoch > 99:
            plt.plot(global_loss_g)
            plt.savefig(str(epoch) + " epoch_g.jpg")
            plt.plot(global_loss_d)
            plt.savefig(str(epoch) + " epoch_d.jpg")

    plt.plot(global_loss_g)
    plt.savefig("final loss_g.jpg")
    plt.plot(global_loss_d)
    plt.savefig("final loss_d.jpg")


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    cuda_id = 0
    # set 8 gpu
    if args.cuda != None:
        if args.cuda == 'cuda0':
            cuda_id = 0
        elif args.cuda == 'cuda1':
            cuda_id = 1
        elif args.cuda == 'cuda2':
            cuda_id = 2
        elif args.cuda == 'cuda3':
            cuda_id = 3
        elif args.cuda == 'cuda4':
            cuda_id = 4
        elif args.cuda == 'cuda5':
            cuda_id = 5
        elif args.cuda == 'cuda6':
            cuda_id = 6
        elif args.cuda == 'cuda7':
            cuda_id = 7

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    #     model = net.Net(params).cuda() if params.cuda else net.Net(params)
    netG = net.Generator(4).cuda(cuda_id)
    netD = net.Discriminator().cuda(cuda_id)

    optimG = optim.Adam(netG.parameters(), lr=params.learning_rate)
    optimD = optim.Adam(netD.parameters(), lr=params.learning_rate)
    if args.optim == 'sgd':
        optimG = optim.SGD(netG.parameters(), lr=params.learning_rate, momentum=0.9)
        optimD = optim.SGD(netD.parameters(), lr=params.learning_rate, momentum=0.9)

    # fetch loss function and metrics
    #     loss_fn = net.loss_fn
    loss_fn = None
    if net == GAN:
        loss_fn = GeneratorLoss().cuda(cuda_id)

    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(netG, netD, train_dl, val_dl, optimG, optimD, loss_fn, metrics, params, args.model_dir,
                       args.restore_file, cuda_id=cuda_id)

    print("finish training and evaluating!")
