import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset.CamVid import CamVid
from dataset.IDDA import IDDA
from loss import DiceLoss, loss_calc
from model.build_BiSeNet import BiSeNet
from model.discriminator import Discriminator
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu, adjust_learning_rate, cal_miou


# noinspection DuplicatedCode
def val(args, model, dataloader, csv_path):
    print("\n", "=" * 100, sep="")
    print('Start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()

        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))

        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy

            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print(f'precision per pixel for test: {precision:.3f}')
        print(f'mIoU for validation: {miou:.3f}')

        print(miou_dict)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)

        print("=" * 100, "\n", sep="")

        return precision, miou


# noinspection DuplicatedCode
def train(args, model, model_D, optimizer, optimizer_D, dataloader_train_S,
          dataloader_train_T, dataloader_val, csv_path, curr_epoch):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    scaler = GradScaler()

    # BisNet loss
    if args.loss == 'dice':
        loss_func = DiceLoss()

    # Discriminator loss
    bce_loss = torch.nn.BCEWithLogitsLoss()

    max_miou = 0
    step = 0

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for epoch in range(curr_epoch + 1, args.num_epochs + 1):

        adjust_learning_rate(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        adjust_learning_rate(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)
        lr_seg = optimizer.param_groups[0]['lr']
        lr_D = optimizer_D.param_groups[0]['lr']

        model.train()
        model_D.train()

        tq = tqdm.tqdm(total=len(dataloader_train_T) * args.batch_size)
        tq.set_description(f'epoch {epoch}, lr_seg {lr_seg:.6f}, lr_D {lr_D:.6f}')
        loss_seg_record = []
        loss_adv_record = []
        loss_D_record = []

        sourceloader_iter = enumerate(dataloader_train_S)
        targetloader_iter = enumerate(dataloader_train_T)
        S_size = len(dataloader_train_S)
        T_size = len(dataloader_train_T)

        for i in range(T_size):
            # -----------------------------------------------------------------------------------------------------------
            # train G (segmentation network)
            # -----------------------------------------------------------------------------------------------------------
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with SOURCE ***********************************************
            _, batch = next(sourceloader_iter)
            data, label = batch
            data = data.cuda()
            label = label.long().cuda()

            with autocast():
                pred, output_sup1, output_sup2 = model(data)

                if args.loss == 'dice':
                    loss_seg1 = loss_func(pred, label)
                    loss_seg2 = loss_func(output_sup1, label)
                    loss_seg3 = loss_func(output_sup2, label)
                    loss_seg = loss_seg1 + loss_seg2 + loss_seg3

                elif args.loss == 'crossentropy':
                    loss_seg1 = loss_calc(pred, label)
                    loss_seg2 = loss_calc(output_sup1, label)
                    loss_seg3 = loss_calc(output_sup2, label)
                    loss_seg = loss_seg1 + loss_seg2 + loss_seg3

            scaler.scale(loss_seg).backward()

            # train with TARGET ***********************************************
            _, batch = next(targetloader_iter)
            data, _ = batch
            data = data.cuda()

            with autocast():
                pred_target, _, _ = model(data)

                D_out = model_D(F.softmax(pred_target))

                loss_adv_target = bce_loss(D_out,
                                           Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

                loss_adv = args.lambda_adv_target * loss_adv_target

            scaler.scale(loss_adv).backward()

            # -----------------------------------------------------------------------------------------------------------
            # train D (discriminator)
            # -----------------------------------------------------------------------------------------------------------

            for param in model_D.parameters():
                param.requires_grad = True

            # train with SOURCE ***********************************************
            pred = pred.detach()

            with autocast():
                D_out = model_D(F.softmax(pred))

                loss_D = bce_loss(D_out,
                                  Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())

                loss_D = loss_D / 2

            scaler.scale(loss_D).backward()

            # train with TARGET ***********************************************
            pred_target = pred_target.detach()

            with autocast():
                D_out = model_D(F.softmax(pred_target))

                loss_D = bce_loss(D_out,
                                  Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())

                loss_D = loss_D / 2

            scaler.scale(loss_D).backward()

            tq.update(args.batch_size)
            losses = {"loss_seg": '%.6f' % loss_seg,
                      "loss_adv": '%.6f' % (loss_adv.item()),
                      "loss_D": '%.6f' % (loss_D.item())}
            tq.set_postfix(losses)

            scaler.step(optimizer)
            scaler.step(optimizer_D)

            step += 1

            writer.add_scalar('loss_seg_step', loss_seg, step)
            writer.add_scalar('loss_adv_step', loss_adv, step)
            writer.add_scalar('loss_D_step', loss_D, step)
            loss_seg_record.append(loss_seg.item())
            loss_adv_record.append(loss_adv.item())
            loss_D_record.append(loss_D.item())

            scaler.update()
        tq.close()

        loss_seg_mean = np.mean(loss_seg_record)
        loss_adv_mean = np.mean(loss_adv_record)
        loss_D_mean = np.mean(loss_D_record)
        writer.add_scalar('epoch/loss_epoch_seg', float(loss_seg_mean), epoch)
        writer.add_scalar('epoch/loss_epoch_adv', float(loss_adv_mean), epoch)
        writer.add_scalar('epoch/loss_epoch_D', float(loss_D_mean), epoch)
        print(f'loss for segmentation : {loss_seg_mean:.6f}')
        print(f'loss for adversarial : {loss_adv_mean:.6f}')
        print(f'loss for discriminator : {loss_D_mean:.6f}')

        # **** Checkpoint saving ****
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            print("\n", "*" * 100, sep="")
            print("Saving checkpoint...")
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'model_D_state_dict': model_D.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
            }

            torch.save(checkpoint,
                       os.path.join(args.save_model_path, 'latest_DA_model_checkpoint.pth'))
            print("Done!")
            print("*" * 100, "\n", sep="")
        #
        # **** Validation model saving ****
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, csv_path)
            if miou > max_miou:
                max_miou = miou
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


# noinspection DuplicatedCode
def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=100, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--learning_rate_D', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--dataCamVid', type=str, default='', help='path of training data')
    parser.add_argument('--dataIDDA', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument("--lambda_adv_target", type=float, default=0.001, help="lambda_adv for adversarial training.")

    args = parser.parse_args(params)
    print(params)

    # create dataset and dataloaderv for CamVid
    train_path = [os.path.join(args.dataCamVid, 'train'), os.path.join(args.dataCamVid, 'val')]
    train_label_path = [os.path.join(args.dataCamVid, 'train_labels'), os.path.join(args.dataCamVid, 'val_labels')]
    test_path = os.path.join(args.dataCamVid, 'test')
    test_label_path = os.path.join(args.dataCamVid, 'test_labels')
    csv_path = os.path.join(args.dataCamVid, 'class_dict.csv')
    dataset_train_T = CamVid(train_path,
                             train_label_path,
                             csv_path,
                             scale=(args.crop_height, args.crop_width),
                             loss=args.loss,
                             mode='train')

    dataloader_train_T = DataLoader(dataset_train_T,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=True)
    dataset_val = CamVid(test_path,
                         test_label_path,
                         csv_path,
                         scale=(args.crop_height, args.crop_width),
                         loss=args.loss,
                         mode='test')

    dataloader_val = DataLoader(dataset_val,
                                # this has to be 1
                                batch_size=1,
                                shuffle=True,
                                num_workers=args.num_workers)

    # create dataset and dataloaderv for IDDA
    train_path = os.path.join(args.dataIDDA, 'rgb')
    train_label_path = os.path.join(args.dataIDDA, 'labels')
    json_path = os.path.join(args.dataIDDA, 'classes_info.json')

    dataset_train_S = IDDA(train_path,
                           train_label_path,
                           json_path,
                           scale=(720, 1280),
                           crop=(args.crop_height, args.crop_width),
                           loss=args.loss)

    dataloader_train_S = DataLoader(dataset_train_S,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=True)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    model_D = Discriminator(in_channels=args.num_classes)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_D = torch.nn.DataParallel(model_D).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    # load pretrained model if exists
    curr_epoch = 0
    if args.pretrained_model_path is not None:
        print("\n", "*" * 100, sep="")
        print(f'Loading model from {args.pretrained_model_path} ...')

        loaded_checkpoint = torch.load(args.pretrained_model_path)
        model.module.load_state_dict(loaded_checkpoint['model_state_dict'])
        model_D.module.load_state_dict(loaded_checkpoint['model_D_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        optimizer_D.load_state_dict(loaded_checkpoint['optimizer_D_state_dict'])
        curr_epoch = loaded_checkpoint['epoch'] + 1

        print(f"\t- epoch done from last checkpoint: {curr_epoch - 1}")
        print('Done!')
        print("*" * 100, "\n", sep="")

    # train
    train(args, model, model_D, optimizer, optimizer_D, dataloader_train_S, dataloader_train_T, dataloader_val,
          csv_path, curr_epoch)

    val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--learning_rate_D', '1e-4',
        '--dataCamVid', '../datasets/CamVid/',
        '--dataIDDA', '../datasets/IDDA/',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_DA',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        # '--pretrained_model_path', './checkpoints_DA/latest_DA_model_checkpoint.pth',
        '--checkpoint_step', '2',
        '--loss', 'crossentropy',
    ]
    main(params)
