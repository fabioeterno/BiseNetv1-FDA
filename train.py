import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
from dataset.IDDA import IDDA, colorize_mask  
#from AdaptSeg
import torch.nn.functional as F
from model.discriminator import FCDiscriminator
from torch.autograd import Variable
#from utils.loss import CrossEntropy2d


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(args , optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_epochs, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval() # I set the model with the evaluation settings
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            #print(predict.shape)
            #print(predict[0].cpu().numpy().permute(1,2,0).shape)
            #predict = reverse_one_hot(predict[0].cpu().numpy())
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

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)[:-1]
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou


def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, curr_epoch, model_D1, optimizer_D1, interp, interp_target, bce_loss):
    import os
    print("Folder to save already present? : " + str(os.path.isdir(args.save_model_path)))
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss()
    max_miou = 0
    step = 0
    source_label = 0
    target_label = 1
    targetloader_iter = enumerate(dataloader_source)

    scaler = amp.GradScaler()


    for epoch in range(curr_epoch, args.num_epochs):
        # Learning rate of the model, default value 0.02500, but if a pretrained model is recovered is adapted
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train() # Setting the model in training mode
        tq = tqdm.tqdm(total=len(dataloader_source))
        tq.set_description('epoch %d, lr %f' % (epoch, lr)) # Print epoch and learning rate 
        loss_record = []
        targetloader_iter = enumerate(dataloader_target)
        sourceloader_iter = enumerate(dataloader_source)

        print("")
        print("target lenght: " + str(len(dataloader_target)) + "source lenght: " + str(len(dataloader_source)) )

        iterations = len(dataloader_target)

        for i in range(iterations):
        #for i in range(2):


            adjust_learning_rate_D(args, optimizer, epoch)
            adjust_learning_rate_D(args, optimizer_D1, epoch)
            optimizer.zero_grad()
            optimizer_D1.zero_grad()

            _, batch = next(sourceloader_iter)
            data, label = batch
            #print("data " + str(data.shape) + "  " + " label: " + str(label.shape))
            
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # train G

            for param in model_D1.parameters():
                param.requires_grad = False

            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            #Train G with source
            with amp.autocast():
                output, output_sup1, output_sup2 = model(data)

                loss1 = loss_func(output, label)
                loss2 = loss_func(output_sup1, label)
                loss3 = loss_func(output_sup2, label)
                loss = loss1 + loss2 + loss3

            #print("")
            #print("output.shape " + str(output.shape))
            #print("output_sup1.shape " + str(output_sup1.shape))
            #print("output_sup2.shape " + str(output_sup2.shape))

            #output = interp(output)
            #output_sup1 = interp(output_sup1)
            #output_sup2 = interp(output_sup2)

            #print("loss1.shape " + str(loss1))
            #print("loss2.shape " + str(loss2))
            #print("loss3.shape " + str(loss3))
            #print("loss1 + loss2 + loss3  " + str(loss))

            # proper normalization
            #loss = loss / args.iter_size
            scaler.scale(loss).backward()

            # Add segmentation loss
            loss_seg_value1 += loss1.data.cpu().numpy() + loss2.data.cpu().numpy() + loss3.data.cpu().numpy()

            # train with target
            
            # try:
            #     _, batch = next(targetloader_iter)
            # except :
            #     targetloader_iter = enumerate(dataloader_target)
            #     _, batch = next(targetloader_iter)

            _, batch = next(targetloader_iter)

            images, _ = batch
            images = images.cuda()
            #print("+----------------------------------+")
            #print("+--------- New Iteration ----------+")
            #print("+----------------------------------+")
            #print("images shape from target" + str(images.shape))

            with amp.autocast():
                pred_target1, _, _ = model(images)
                pred_target1 = interp_target(pred_target1)

                D_out1 = model_D1(F.softmax(pred_target1))

                loss_adv_target1 = bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            
                loss = args.lambda_adv_target1 * loss_adv_target1
                loss = loss / args.iter_size                #Should be noticed
            scaler.scale(loss).backward()

            #print("+-------- Generator phase ---------+")
            #print("Iteration : " + str(i))
            #print("Images shape: " + str(images.shape) + " - Prediction shape :" + str(pred_target1.shape))
            #print("+-------- Discriminator prediction ------------+")
            #print("D_out1 shape : " + str(D_out1.shape) + " - D_out1 : " + str(D_out1))
            #print("+-------- Adversarial loss ---------+")
            #print("args.lambda_adv_target1 : " + str(args.lambda_adv_target1) + " - loss_adv_target1 : " + str(loss_adv_target1))
            #print("+-------- Predictions target ------------+")
            #print("pred_target1  " + str(pred_target1) + " - " + "pred_target1 shape" + str(pred_target1.shape))
            #print("\n los adv target=",loss_adv_target1.data.cpu().numpy())
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size #Should be noticed 

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True
            
            # train with source
            pred1 = output.detach()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(pred1))
            
                loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

                loss_D1 = loss_D1 / args.iter_size / 2
            scaler.scale(loss_D1).backward()
            loss_D_value1 += loss_D1.data.cpu().numpy()
            
            # train with target
            pred_target1 = pred_target1.detach()
            with amp.autocast():
                D_out1 = model_D1(F.softmax(pred_target1))
                loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
                loss_D1 = loss_D1 / args.iter_size / 2
            scaler.scale(loss_D1).backward()
            loss_D_value1 += loss_D1.data.cpu().numpy()

            #print("")
            #print("+-------- Discriminator phase ---------+")
            #print("+-------- Predictions source------------+")
            #print("pred1 source   " + str(pred1) + " - " + "pred1 source shape" + str(pred1.shape))
            #print("+-------- Predictions target ------------+")
            #print("pred1 target   " + str(pred_target1) + " - " + "pred1 target shape" + str(pred_target1.shape))
            #print("+-------- Discriminator loss ------------+")
            #print("loss_D_value1 shape : " + str(loss_D_value1.shape) + " - loss_D_value1 : " + str(loss_D_value1))

            print('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_adv1 = {3:.3f}, loss_D1 = {4:.3f}'.format(
                i, iterations, loss_seg_value1, loss_adv_target_value1, loss_D_value1))

            scaler.step(optimizer)
            scaler.step(optimizer_D1)
            scaler.update()

        loss_total = loss_seg_value1
        tq.update(args.batch_size)
        tq.set_postfix(loss='%.6f' % loss_total)
        step += 1
        writer.add_scalar('loss_step', loss_total, step)
        loss_record.append(loss_total.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0: # and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                print("making directory " + args.save_model_path)
                os.mkdir(args.save_model_path)
            # Saving in the checkpoint also the epoch number and the optimizer to resume them --> FE
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_D1': optimizer_D1.state_dict()
            }

            #torch.save(model.module.state_dict(),os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
            torch.save(checkpoint, os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
            print("saving the model " + args.save_model_path)

        if epoch % args.validation_step == 0: #and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                # Saving the same type of checkpoint of the training step
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_D1': optimizer_D1.state_dict()
                }
                torch.save(model.module.state_dict(),os.path.join(args.save_model_path, 'best_dice_loss.pth'))
                torch.save(checkpoint, os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


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
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')

    #discriminator arguments
    parser.add_argument("--target", type=str, default='cityscapes', help="available options : cityscapes") #change the variable
    parser.add_argument("--ignore-label", type=int, default = 255, help="The index of the label to ignore during the training.")
    parser.add_argument("--lambda-seg", type=float, default = 0.1, help="lambda_seg")
    parser.add_argument("--lambda-adv-target1", type=float, default = 0.0002, help = "lambda adv for adversarial training ")
    parser.add_argument("--lambda-adv-target2", type=float, default = 0.001, help = "lambda adv for adversarial training ")
    parser.add_argument("--momentum", type=float, default = 0.9, help="Momentum component of the optimiser")
    parser.add_argument("--learning-rate-D", type=float, default=1e-4, help="Base learning rate for discriminator")
    parser.add_argument("--gan", type=str, default='Vanilla', help="choose the GAN objective.") #I dont understand this shit
    parser.add_argument("--iter-size", type=int, default=1, help="iteration size")

    args = parser.parse_args(params)
    # print(os.path.isdir(args.save_model_path)) # --> False

    # create dataset and dataloader
    # Train and label path
    target_path = [os.path.join(args.data, 'train'), os.path.join(args.data, 'val'), os.path.join(args.data, 'test')]
    target_label_path = [os.path.join(args.data, 'train_labels'), os.path.join(args.data, 'val_labels'), os.path.join(args.data, 'test_labels')]
    # Test and label path




    
    csv_path = os.path.join(args.data, 'class_dict.csv')



    # Defining dataset training, I crop with the height and width of images passed as parameters
    # and I pass the defined loss
    # dataset_train = CamVid(train_path, test_label_path, csv_path, scale=(args.crop_height, args.crop_width),
    #                        loss=args.loss, mode='train')
    # # Dataloader for Training set


    dataset_target = CamVid(target_path, target_label_path, csv_path, scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='train')
    
    dataloader_target = DataLoader(
        dataset_target,
        # this has to be 1
        batch_size = 2,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    dataset_source = IDDA()

    dataloader_source = DataLoader(
        dataset_source,
        batch_size = 2,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_path = os.path.join(args.data, 'test')
    val_labels_path = [os.path.join(args.data, 'test_labels')]
    dataset_val = CamVid(val_path, val_labels_path, csv_path, scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='test')

    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    trainloader_iter = enumerate(dataloader_source)
    targetloader_iter = enumerate(dataloader_target)

    # build model for generator
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()


    #build model for discriminator
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1.train() #initialize
    #model_D1.cuda(args.cuda)
    if torch.cuda.is_available() and args.use_gpu:
        model_D1 = torch.nn.DataParallel(model_D1).cuda()
    


    # build optimizer for generator
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None



    #build optimizer for discriminator
    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    
    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    
    interp = torch.nn.Upsample(size=(960,720), mode='bilinear')
    interp_target = torch.nn.Upsample(size=(960,720), mode='bilinear')


    # load pretrained model if exists
    curr_epoch = 0
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        checkpoint = torch.load(args.pretrained_model_path)
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
        print("epoch trained from checkpoint: " + str(checkpoint['epoch']))
        curr_epoch = checkpoint['epoch']
        print('Done!')
      
      
    # train (comment validation)
    train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, curr_epoch, model_D1, optimizer_D1, interp, interp_target, bce_loss)

    # validation (comment training)
    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '101',
        '--checkpoint_step', '2',
        '--validation_step', '2',
        '--learning_rate', '2.5e-2',
        '--data', '/content/drive/MyDrive/Politecnico/Machine Learning/BiseNetv1/dataset/CamVid',
        '--num_workers', '8',
        '--num_classes', '12',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd', # we can try adam
        '--pretrained_model_path', './checkpoints_101_sgd/latest_dice_loss.pth'
    ]
    main(params)

