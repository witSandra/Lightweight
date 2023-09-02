import os
import cv2
import numpy as np
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.LD2_BS import FastDepth    # 导入不同模块
import utils.metrics as metrics
from utils.criterion import SiLogLoss
import utils.my_logging as logging

from datasets.base_dataset import get_dataset
from configs.train_options import TrainOptions
from torchstat import stat
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

import time

def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Logging
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')
    log_loss = os.path.join(log_dir, 'loss.txt')
    logging.log_args_to_txt(log_txt, args)
    logging.log_args_to_txt(log_loss, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = FastDepth(max_depth=args.max_depth)

    # FLOPs and parameters
    # stat(model, (3, 448, 576))
    # stat(model, (3, 352, 1216))

    # Resume = True
    # if Resume:
    #     path = "/mnt/harddisk1/Duanxiuzhen/codespace/FastDepth/logs/kitti/0524_lightNet_BRCshuff8_eca/epoch_25_model.ckpt"
    #     model = torch.load(path)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    if args.dataset == 'nyudepthv2':
        dataset_kwargs['crop_size'] = (448, 576)
    elif args.dataset == 'kitti':
        dataset_kwargs['crop_size'] = (352, 704)    # 352, 704
    else:
        dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               shuffle=True, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    global global_step
    global_step = 0

    t1 = time.perf_counter()        # 记录训练开始时间
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    enumerate
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        writer.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                              device=device, epoch=epoch, args=args,
                                              log_dir=log_dir)
            writer.add_scalar('Val loss', loss_val, epoch)

            result_lines = logging.display_result(results_dict)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)
        scheduler.step()  # 一次epoch对应一次scheduler.step()，进行学习率更新

    print('Time(h)：', (time.perf_counter() - t1) / 3600)

    with open(log_loss, 'a') as txtfile:
        txtfile.write('\nTime:' + str((time.perf_counter() - t1) / 3600))


def train(train_loader, model, criterion_d, optimizer, device, epoch, args):
    global global_step
    model.train()

    depth_loss = logging.AverageMeter()
    exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    log_loss = os.path.join(log_dir, 'loss.txt')
    logging.log_args_to_txt(log_loss, args)

    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1
        # ReduceLROnPlateau
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)

        preds = model(input_RGB)        #

        optimizer.zero_grad()           # (1)
        loss_d = criterion_d(preds['pred_d'].squeeze(), depth_gt)
        
        depth_loss.update(loss_d.item(), input_RGB.size(0))
        loss_d.backward()               # (2)

        logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %.4f (%.4f)' %
                            (depth_loss.val, depth_loss.avg)))      # loss

        optimizer.step()                # (3)


    with open(log_loss, 'a') as txtfile:
        txtfile.write("\n第%d个lr：%f, Depth Loss: %.4f (%.4f)" % (epoch, optimizer.state_dict()['param_groups'][0]['lr'],depth_loss.val, depth_loss.avg))
        # txtfile.write('\nDepth Loss: %.4f (%.4f)' % (depth_loss.val, depth_loss.avg))

    return loss_d


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir):
    depth_loss = logging.AverageMeter()
    model.eval()

    if args.save_model:
        # torch.save(model.state_dict(), os.path.join(
        #     log_dir, 'epoch_%02d_model.ckpt' % epoch))
        torch.save(model, os.path.join(
            log_dir, 'epoch_%02d_model.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]

        with torch.no_grad():
            # print("input_RGB:", input_RGB.size())    #input_RGB:torch.Size([1, 3, 480, 640])
            preds = model(input_RGB)

        pred_d = preds['pred_d'].squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion_d(preds['pred_d'].squeeze(), depth_gt)

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)   ###########
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        save_path = os.path.join(result_dir, filename)

        if save_path.split('.')[-1] == 'jpg':
            save_path = save_path.replace('jpg', 'png')

        if args.save_result:
            if args.dataset == 'kitti':
                pred_d_numpy = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d_numpy = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, loss_d


if __name__ == '__main__':
    main()
