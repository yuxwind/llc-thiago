#!/usr/bin/python3

"""
    Sample Runs:
    Training a model
    ./train_fcnn.py --arch fcnn2f --save-dir models/fcnn_run_112/ -v | tee models/fcnn_run_112/train.log
    ./train_fcnn.py --arch fcnn2c_d --epochs 61 --save-dir models/fcnn_run_32/ | tee models/fcnn_run_32/run_32.log

    Evaluating a model
    ./train_fcnn.py --arch fcnn2c_d --resume models/fcnn_run_32/checkpoint_60.tar -e

    Master script to evaluate the performance of fixing augmentation, activation pattern, adversarial training and other things.

    Version 6 Xin Yu        2021-05~08     
        1. Support MNIST, Cifar10-gray, Cifar10-rgb, Cifar100. Code for datasets is moved to dataset.py
        2. Support training/evaluating LeNet
        3. Support evaluation on pruned FCNN
        4. Support evaluation on pruned LeNet
        5. Get stable neurons of FCNN from training/test dataset
        6. Get stable neurons of Lenet from training/test dataset
    Version 5 Abhinav Kumar 2018-03-25 Code cleaned, augmentation removed in MNIST
    Version 4 Abhinav Kumar 2018-03-15 L1 regularisation added
    Version 3 Abhinav Kumar 2018-03-15 Buggy optimizer fixed
    Version 2 Abhinav Kumar 2018-03-11 PGD added
    Version 1 Abhinav Kumar 2018-11-10
"""

import argparse
import os
import shutil
import time
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, './train')
import fcnn
from utils import DataParallelPassThrough
from dataset import get_data

#import VGG
import torch.nn.functional as F
import numpy as np

model_names = sorted(name for name in fcnn.__dict__
    if name.islower() and not name.startswith("__")
                     and (name.startswith("fcnn") or name == 'lenet')
                     and callable(fcnn.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch MNIST/CFAR-10 Training')
parser.add_argument( '--arch', metavar='ARCH', default='fcnn1',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: fcnn1)')

parser.add_argument('-j'  , '--workers'        , type=int        , default=2             , metavar='N'    , help='number of data loading workers (default: 4)')
parser.add_argument('-b'  , '--batch-size'     , type=int        , default=128            , metavar='N'    , help='mini-batch size (default: 64)')
parser.add_argument(        '--test-batch-size', type=int        , default=10000         , metavar='T'    , help='input batch size for testing (default: 10000)')

# Optimiser Options
parser.add_argument(        '--epochs'         , type=int         , default=121          , metavar='N'    , help='number of total epochs to run (default:121)')
parser.add_argument('--lr', '--learning_rate'  , type=float       , default=0.01         , metavar='LR'   , help='initial learning rate (default: 0.01')
parser.add_argument(        '--momentum'       , type=float       , default=0.9          , metavar='M'    , help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight_decay'   , type=float       , default=0.00         , metavar='W'    , help='weight decay  (default: 0.00)')
parser.add_argument('--l1', '--l1_regular'     , type=float       , default=0.00                          , help='l1 regulariser on the first layer (default: 0.00)')
parser.add_argument(        '--print-freq'     , type=int         , default=1            , metavar='N'    , help='print frequency (default: 1)')
parser.add_argument(        '--save-dir'       , type=str         , default='models'     , dest='save_dir', help='The directory used to save the trained models')

# Evaluation Options
parser.add_argument(        '--resume'         , type=str          , default=''          , metavar='PATH' , help='path to latest checkpoint (default: none)')
parser.add_argument(        '--start-epoch'    , type=int          , default=0           , metavar='N'    , help='manual epoch number (useful on restarts)')
parser.add_argument(        '--pretrained'     , dest='pretrained' , action='store_true'                  , help='use pre-trained model')
parser.add_argument('-e'  , '--evaluate'       , dest='evaluate'   , action='store_true'                  , help='evaluate model on validation set')
parser.add_argument( '--eval-stable'           , dest='eval_stable', action='store_true'                  , help='evaluate the sbability of neurons')
parser.add_argument('--eval-train-data'    , action='store_true', help='evaluate model on training set')
parser.add_argument(        '--half'           , dest='half'       , action='store_true'                  , help='use half-precision(16-bit) ')
parser.add_argument(        '--dataset'        , dest='dataset', type=str         , default='MNIST'     , help='Dataset to be used (default: MNIST)')

# More advanced options
parser.add_argument('-a', '--augmentation'     , dest='augmentation', action='store_true'                 , help='use augmentation (default=False)')
parser.add_argument('-s', '--std_dev'          , type=float         , default=0.05                        , help='if use augmentation, then standard value of noise (default=0.05)')
parser.add_argument('-v', '--adversarial'      , dest='adversarial' , action='store_true'                 , help='use adversarial attacks (default=False)')
parser.add_argument('-f', '--fix_activations'                       , default=False                       , help='In case this is set to true, use file from activations_path')
parser.add_argument(      '--beta'             , type=float         , default=0.003      , metavar='be'   , help='tradeoff between softmax and MSE (default: 0.003)')
parser.add_argument('-p', '--activation_path'  , type=str           , default='../prior/class_10_dim_512_sp_0.7_fp_0.5.config', help='fixed_Activations_file_path in the form of csv file')

## imbalanced training dataset options
#parser.add_argument(      '--im_clsid'             , type=int         , default=3     ,help='reduce the data size in im_clsid')
#parser.add_argument(      '--im_ratio'             , type=float         , default=1.  ,help='keep im_ratio of certain class to generate biased training datasest')

args = parser.parse_args()
#ImbalanceCfg = namedtuple('Imbalance', ['clsid', 'keep_ratio'])
#imbalance_cfg = ImbalanceCfg(, 1/256.0) # 1/4, 1/16, 1/64, 1/256

best_prec1 = 0
adversarial_epsilon       = 0.15
adversarial_step_size     = 0.1
adversarial_iterations    = 200
lr_decay_step_size        = 50
lr_decay_gamma            = 0.1
lr_decay_step_size        = 30
lr_decay_gamma            = 0.5
dataset                   = "MNIST"#"CIFAR10" #"MNIST"


################################################################################
# main function
################################################################################
def main():
    global args, best_prec1

    dataset = args.dataset 

    cudnn.benchmark = True
    use_cuda        = True # True
    device = torch.device("cuda" if use_cuda else "cpu")
    fcnn_flag       = True
    sep             = "," #This is used for separating in weights.dat file
    save_frequency  = 120
    eval_frequency  = 30
    #save_frequency  = 1 # DEBUG


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if(dataset == "CIFAR10-gray"):
        input_dim  = 1024
        class_num  = 10
    elif (dataset == "MNIST"):
        input_dim  = 784
        class_num  = 10
    elif (dataset == "CIFAR10-rgb"):
        input_dim  = 1024 * 3
        class_num  = 10
    elif (dataset == "CIFAR100-rgb"):
        input_dim  = 1024 * 3
        class_num  = 100
    
    if args.arch == "lenet":
        fcnn_flag = False
        if("CIFAR10" in dataset):
            input_dim  = 32 
        elif (dataset == "MNIST"):
            input_dim = 28
    
    if args.arch == 'fcnn_prune':
        cfg = get_net_width(os.path.dirname(args.resume))
        model = fcnn.__dict__[args.arch](cfg, input_dim, class_num)
    else:
        model = fcnn.__dict__[args.arch](input_dim, class_num)
        
    if not fcnn_flag: # for lenet
        lenet_prune = fcnn.__dict__['lenetPrune'](input_dim, class_num)
    #model.features = torch.nn.DataParallel(model.features)
    

    try:
        model.features = torch.nn.DataParallel(model.features)
    except:
        pass
    try:
        model.conv_features = torch.nn.DataParallel(model.conv_features)
    except:
        pass

    # define loss function (criterion)
    # If there is no softmax, use CrossEntropyLoss()
    # If there is softmax   , use NLLLoss()
    # https://pytorch.org/docs/stable/nn.html#crossentropyloss
    # criterion = nn.CrossEntropyLoss()
    criterion   = nn.NLLLoss()

    if not args.evaluate:
        print("===============================================================================");
        print("Model :")
        print(model)
        print("\nCriterion :")
        print(criterion)
        print("\n===============================================================================");
    
    def map_lenet_2_lenetmask(m1, m2):
        m1.conv1.weight.data = m2.conv_features.module[0].weight.data
        m1.conv1.bias.data = m2.conv_features.module[0].bias.data
        
        m1.conv2.weight.data = m2.conv_features.module[3].weight.data
        m1.conv2.bias.data = m2.conv_features.module[3].bias.data

        m1.fc1.weight.data = m2.features.module[0].weight.data
        m1.fc1.bias.data = m2.features.module[0].bias.data

        m1.fc2.weight.data = m2.features.module[2].weight.data
        m1.fc2.bias.data = m2.features.module[2].bias.data

        m1.fc3.weight.data = m2.classifier[0].weight.data
        m1.fc3.bias.data = m2.classifier[0].bias.data

        return m1, m2

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            
            if not fcnn_flag:
                map_lenet_2_lenetmask(lenet_prune, model)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Transfer model and criterion to default device
    model = model.to(device)
    if not fcnn_flag:
        lenet_prune = lenet_prune.to(device)
    criterion = criterion.to(device)

    stddev = args.std_dev
    distbn = torch.distributions.normal.Normal(0, stddev)

    normalize  = transforms.Normalize(mean=[0], std=[1])

    if(args.fix_activations):
        print("Beta = " + str(args.beta))
        print("Using preactivations file " + str(args.activation_path))
    else:
        args.beta = 0
        print("Beta = 0")


    if args.evaluate:
        # For evaluation mode

        if args.adversarial:
            print("Carrying out adversarial attacks")
            epsilons   = [0, .05, .1, .15, .2, .25, .3]
            accuracies = []
            examples   = []
            val_loader = get_data(dataset, args.test_batch_size, args.workers, train=False, shuffle=False)

            # Run test for each epsilon
            for eps in epsilons:
                acc, ex = test_adversarial(model, criterion, device, val_loader, eps)
                accuracies.append(acc)
                examples.append(ex)
            return
        else:
            # Consider using augmentation: Std deviation of the noise while testing/evaluation = stddev
            if not args.eval_stable: 
                val_loader = get_data(dataset, args.test_batch_size, args.workers, 
                                    train=False, shuffle=False, 
                                    has_augmentation = args.augmentation)
                acc, all_pred, all_correct, all_output = validate(val_loader, model, criterion, 1, device, fcnn_flag)
                if args.resume and 'magnitude_pruned_' in args.resume:
                    orig_model = args.resume.replace('magnitude_pruned_', '')
                    print("=> loading checkpoint '{}'".format(orig_model))
                    orig_ckp = torch.load(orig_model)
                    model.load_state_dict(orig_ckp['state_dict'])
                    orig_acc, orig_all_pred, orig_all_correct, orig_all_output = validate(
                            val_loader, model, criterion, 1, device, fcnn_flag)
                    diff_pred = (all_pred != orig_all_pred).sum()
                    diff_output = (orig_all_output - all_output).abs().max()
                    diff_output_cls = (orig_all_output.max(dim=1)[0] - all_output.max(dim=1)[0]
                                        ).abs().max()
                print('datasize:', len(val_loader.dataset))
                print('acc:', acc)
                fpath = os.path.join(os.path.dirname(args.resume), 'eval.txt')
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        l = f.readlines()[-1].strip().split(',')
                        if len(l) == 4:
                            acc_orig, acc_llc, acc_mc = l[1], l[2], l[3]
                        if len(l) == 7:
                            acc_orig, acc_llc, acc_mc, diff_pred_orig_mc, diff_out_orig_mc, \
                                    diff_out_cls_orig_mc = l[1], l[2], l[3], l[4], l[5], l[6]

                else:
                    acc_orig = '-'
                    acc_llc  = '-'
                    acc_mc   = '-'
                    diff_pred_orig_mc, diff_out_orig_mc, diff_out_cls_orig_mc = '-', '-', '-'
                ckp_name = os.path.basename(args.resume)
                if ckp_name == 'magnitude_pruned_checkpoint_120.tar':
                    acc_mc = acc
                    acc_orig = orig_acc
                    diff_pred_orig_mc = diff_pred
                    diff_out_orig_mc  = diff_output
                    diff_out_cls_orig_mc = diff_output_cls
                elif ckp_name == 'pruned_checkpoint_120.tar':
                    acc_llc = acc
                else:
                    acc_orig = acc
                with open(fpath, 'w') as f:
                    info = args.resume + f',{acc_orig},{acc_llc},{acc_mc},{diff_pred_orig_mc}' + \
                            f',{diff_out_orig_mc},{diff_out_cls_orig_mc}'
                    print(info)
                    f.write(info + '\n')
            else:
                print('load checkpoints: ', args.resume)
                args.eval_train_data = True 
                active_states, acc, masks = eval_active_state(val_loader, model, criterion, fcnn_flag)
                # Get the index of stable neurons, the input is layer 0 and the layer index starts from 1  
                stably_active_ind, stably_inactive_ind = find_stable_neurons(active_states)
                # write the index into the checkpoints' folder
                np.save(os.path.join(os.path.dirname(args.resume), 'stable_neurons.train.npy'), {
                            'stably_active': stably_active_ind.numpy(),
                            'stably_inactive': stably_inactive_ind.numpy()
                })
                # on eval dataset
                args.eval_train_data = False
                val_loader = get_data(dataset, args.test_batch_size, args.workers, 
                                        train=args.eval_train_data, shuffle=False, 
                                        has_augmentation = args.augmentation)
                print('datasize:', len(val_loader.dataset))
                print('load checkpoints: ', args.resume)
                active_states_, acc_, masks_ = eval_active_state(val_loader, model, criterion, fcnn_flag)
                # Get the index of stable neurons, the input is layer 0 and the layer index starts from 1  
                if not fcnn_flag:
                    print('+++++++++++++++++++++++')
                    acc_no,_,_,_ = validate(val_loader, model, criterion, 1, device, fcnn_flag)
                    acc_mask,_,_,_ = validate(val_loader, lenet_prune, criterion, 1, device, 
                            fcnn_flag, masks)
                stably_active_ind_, stably_inactive_ind_ = find_stable_neurons(active_states_)

                def diff_arr(arr1, arr2): # for train, test, Nx2x1 
                    s1=[]
                    s2=[]
                    for i in range(arr1.shape[0]):
                        s1.append(f'{arr1[i,0,0]}-{arr1[i,1,0]}')
                    for i in range(arr2.shape[0]):
                        s2.append(f'{arr2[i,0,0]}-{arr2[i,1,0]}')
                    s1 = set(s1)
                    s2 = set(s2)
                    if len(s1) == 0:
                        wrong_train = 0.0
                    else:
                        wrong_train = len(s1-s2)/len(s1)
                    return wrong_train, len(s1), len(s2)
                    

                wrong_train1, cc_train1, cc_test1 = diff_arr(stably_inactive_ind, 
                        stably_inactive_ind_)
                wrong_train2, cc_train2, cc_test2 = diff_arr(stably_active_ind, stably_active_ind_)
                fpath = os.path.join(os.path.dirname(args.resume), 'preprocesss-train_test.txt')
                print(fpath)
                with open(fpath, 'w') as f:
                    info = args.resume + f', {cc_train2}, {cc_test2}, {wrong_train2:.04f}'  + \
                            f',, {cc_train1}, {cc_test1}, {wrong_train1:.04f}' + \
                            f',, {acc_no:.04f}, {acc_mask:.04f}, {acc_mask-acc_no:.04f}' 
                    print(info)
                    f.write(info + '\n')


                # write the index into the checkpoints' folder
                np.save(os.path.join(os.path.dirname(args.resume), 'stable_neurons.train_test.npy'), {
                            'stably_active.train': stably_active_ind.numpy(),
                            'stably_inactive.train': stably_inactive_ind.numpy(),
                            'stably_active.eval': stably_active_ind_.numpy(),
                            'stably_inactive.eval': stably_inactive_ind_.numpy(),
                            'stably_active.diff':   [ cc_train2, cc_test2, wrong_train2],
                            'stably_inactive.diff': [ cc_train1, cc_test1, wrong_train1]
                })

                
            # Write model weights in CPLEX Format
            #save_weights_in_cplex_format(model, os.path.dirname(args.resume), os.path.basename(args.resume), input_dim, acc, sep)


            # Write tensor to csv file for future use in the same directory as save
            #features_folder       = os.path.dirname(args.resume)


            #print("\nFeatures [{0}x{1}]".format(features.shape[0],features.shape[1]))
            #features_folder      = 'features/'
            #activations_file_name = dataset + args.arch + "_noise_" + str(stddev) + "_smx.txt"
            #activations_file_path = os.path.join(features_folder,activations_file_name)
            #write_tensor_to_csv_file(features, activations_file_path)
            #print("Saved to " + activations_file_path + "\n")

            return


    else:
        #For training mode

        val_loader = get_data(dataset, args.test_batch_size, args.workers, 
                                    train=args.eval_train_data, shuffle=False, 
                                    has_augmentation = False)


        if args.augmentation:
            print("Using augmentation. Std deviation of the noise while training = " + str(stddev))
            transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))
        else:
            print("No data augmentation added in training")

        if args.adversarial:
            print("Using adversarial training with lr = %0.2f, epsilon = %0.2f, iterations = %d" %(adversarial_step_size, adversarial_epsilon, adversarial_iterations))

        print("\nModel save folder  = %s"   %(args.save_dir))
        print("-------------------------------")
        print("Optimisation Parameters")
        print("-------------------------------")
        print("lr                 = %.4f" %(args.lr))
        print("momentum           = %.4f" %(args.momentum))
        print("weight decay       = %.4f" %(args.wd))
        print("l1 regul           = %.5f" %(args.l1))
        print("epochs             = %d"   %(args.epochs))
        print("lr decay step size = %d"   %(lr_decay_step_size))
        print("lr decay gamma     = %.2f" %(lr_decay_gamma))

        print("\n\n")
       
        train_loader = get_data(dataset, args.batch_size, args.workers, train=True, shuffle=True)
        #import pdb;pdb.set_trace()
        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,  weight_decay=args.wd)

        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_gamma)

        if args.half:
            model.half()
            criterion.half()

        # Start training
        for epoch in tqdm(range(args.start_epoch, args.epochs)):
            
            #TODO: adjust for the CIFAR10-rgb, cifar100-rgb 
            #adjust_learning_rate(optimizer, epoch)
            scheduler.step()

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch,  device, fcnn_flag, args)
            # evaluate on validation set
            if (epoch % eval_frequency == 0):
                prec1,_,_,_ = validate(val_loader, model, criterion, epoch, device, fcnn_flag)

                # For MNIST dataset, if accuracy is close to 10%, the model did not converge.
                # Stop training it further. Saves some time especially with adversarial training.
                if (epoch > 0 and dataset == "MNIST" and prec1 < 12):
                    sys.exit("Model DNC!!!")
            if (epoch % save_frequency == 0):
                if args.eval_stable:
                    print('===========eval_stable============')
                    active_states, acc, masks = eval_active_state(train_loader, model, criterion, fcnn_flag)
                    # Get the index of stable neurons, the input is layer 0 and the layer index starts from 1  
                    stably_active_ind, stably_inactive_ind = find_stable_neurons(active_states)
                    # write the index into the checkpoints' folder
                    np.save(os.path.join(args.save_dir, f'stable_neurons.npy'), {
                                'stably_active': stably_active_ind.numpy(),
                                'stably_inactive': stably_inactive_ind.numpy()
                    })

            # remember best prec@1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # save the model and write weights in CPLEX Format
            if (epoch > 0 and epoch % save_frequency == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

                #save_weights_in_cplex_format(model, args.save_dir, "checkpoint_" + str(epoch) +  ".tar", input_dim, prec1, sep)
                save_weights_in_npy_format(model, args.save_dir, 
                        'checkpoint_{}.tar'.format(epoch), input_dim, prec1)


################################################################################
# Run one train epoch
################################################################################
def train(train_loader, model, criterion, optimizer, epoch,  device, fcnn_flag, args):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    adv_losses = AverageMeter()
    top1 = AverageMeter()
    adv_top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (data, target) in enumerate(train_loader):
        #import pdb;pdb.set_trace()
        # measure data loading time
        data_time.update(time.time() - end)
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        if fcnn_flag:
            data = data.view(data.shape[0],-1)

        if args.half:
            data = data.half()

        # Carry out adeversarial training as well
        if args.adversarial:
            # Set requires_grad attribute of tensor. Important for Adversarial Training
            data.requires_grad = True

        # compute output
        output, act = model(data)

        if (args.fix_activations):
            #read csv file
            prior, mask = get_prior_activations(target, args.activation_path)
            prior       = prior.type(act.type())
            mask        = mask.type (act.type())
            loss        = criterion(output, target) + args.beta*F.l1_loss(act*mask, prior*mask) #Check the loss here
        else:
            regularization_loss = 0
            # compute the l1 loss
            # https://discuss.pytorch.org/t/how-to-create-compound-loss-mse-l1-norm-regularization/17171/4
            for m in model.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    regularization_loss += torch.sum(torch.abs(m.weight))
                    #break # for the l1 regularization on the first layer itself break
            #import pdb;pdb.set_trace()
            loss        = criterion(output, target) + args.l1*regularization_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1,_,_ = accuracy(output.data , target)
        prec1 = prec1[0]
        losses.update   (loss.item() , data.size(0))
        top1.update     (prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # Carry out adeversarial training as well
        if args.adversarial:
            # Collect datagrad
            data_grad = data.grad.data

            # *************************************** Check Here ****************************
            # Call FGSM Attack or PGD
            #perturbed_data = fgsm_attack(data, adversarial_epsilon, data_grad)
            perturbed_data =  pgd_attack(data, adversarial_epsilon, data_grad, adversarial_step_size, adversarial_iterations)

            # compute output
            output, act = model(perturbed_data)

            if (args.fix_activations):
                #read csv file
                prior, mask = get_prior_activations(target, args.activation_path)
                prior       = prior.type(act.type())
                mask        = mask.type (act.type())
                # ****************************** Check Loss Here ****************************
                loss        = criterion(output, target) + args.beta*F.l1_loss(act*mask, prior*mask) #Check the loss here
            else:
                regularization_loss = 0
                # compute the l1 loss
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        regularization_loss += torch.sum(torch.abs(m.weight))
                        break # for the l1 regularization on the first layer itself
                loss        = criterion(output, target) + args.l1*regularization_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()
            optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy (output.data , target)[0]
            adv_losses.update(loss.item() , data.size(0))
            adv_top1.update  (prec1.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            """
            if i % args.print_freq == 0:
                print('Epoch Adv: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=adv_losses, top1=adv_top1))
            """
    """
    print('Epoch Gen: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    """


################################################################################
# Run evaluation or validation
################################################################################
def validate(val_loader, model, criterion, epoch, device, fcnn_flag, masks=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_pred = []    # predicted cls id
    all_correct = [] # binary array to indicate correct prediction
    all_output = []      # network output
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        if args.half:
            data = data.half()

        if fcnn_flag:
            data = data.view(data.shape[0],-1)

        # compute output
        if masks is not None:
            output,_ = model(data, masks)
        else:
            output,_ = model(data)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec_, pred_, correct_ = accuracy(output.data , target)
        #import pdb;pdb.set_trace()
        prec1 = prec_[0]
        losses.update   (loss.item() , data.size(0))
        top1.update     (prec1.item(), data.size(0))
        all_pred.append(pred_)      # pred_:    topk x batch_size
        all_correct.append(correct_)# correct_: topk x batch_size
        all_output.append(output)   # output:   batch_size x cls_num

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    all_pred = torch.cat(all_pred, dim=1).permute(1,0)    #data_size x topk 
    all_correct = torch.cat(all_correct, dim=1).permute(1,0) #data_size x topk 
    all_output = torch.cat(all_output, dim=0)             #data_size x cls_num
    print('Epoch: [{0}] * Prec@1 {top1.avg:.3f}'
          .format(epoch, top1=top1))

    return top1.avg, all_pred, all_correct, all_output



################################################################################
# Computes and stores the average and current value
################################################################################
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



################################################################################
# Computes the precision@k for the specified values of k
################################################################################
def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred, correct




################################################################################
# Save the training model
################################################################################
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


################################################################################
# Saves weights and biases in npy format
################################################################################
def save_weights_in_npy_format(model, folder, file_name, input_dim, acc):
    # Writing weights and bias to a file
    npy_file_path = os.path.join(folder,  "weights.npy")
    npy_dict = {}

    # Use a list to store the weights. We donot unravel it here since we will be
    # writing one row in one line in the dat file.
    weights = []
    bias    = []

    params = model.state_dict()
    layers = 0
    hidden = [input_dim]

    for key, value in params.items():
        if ('weight' in  key):
            layers += 1
            if 'conv_features.0.weight' in key:   # for conv1 in lenet
                hidden.append(28*28*6)
            elif 'conv_features.3.weight' in key: # for conv2 in lenet
                hidden.append(10*10*16)
            else:                                 # for fc
                hidden.append(value.shape[0])
            weights.append(value.cpu().detach().numpy())
        if ('bias' in key):
            bias   .append(value.cpu().detach().numpy())

    npy_dict['model_path'] = os.path.join(folder, file_name)
    npy_dict['acc'] = acc
    npy_dict['layers'] = layers
    npy_dict['n'] = hidden
    npy_dict['weight'] = weights
    npy_dict['bias']   = bias

    #import pdb;pdb.set_trace()
    np.save(npy_file_path, npy_dict)

################################################################################
# Saves weights and biases in CPLEX format
################################################################################
def save_weights_in_cplex_format(model, folder, file_name, input_dim, acc, sep):

    # Writing weights and bias to a file
    dat_file_path = os.path.join(folder,  "weights.dat")

    # Use a list to store the weights. We donot unravel it here since we will be
    # writing one row in one line in the dat file.
    weights = []
    bias    = []

    params = model.state_dict()
    layers = 0
    hidden = [input_dim]


    # Python3 uses params.items()
    # Python2 uses params.iteritems()
    for key, value in params.items():
        if ('weight' in  key):
            layers += 1
            hidden.append(value.shape[0])
            weights.append(value.cpu().detach().numpy())
        if ('bias' in key):
            bias   .append(value.cpu().detach().numpy())

    with open(dat_file_path, 'w') as my_file:

        my_file.write("//Model path = " + os.path.join(folder, file_name) + "\n")
        my_file.write("//Classification accuracy = " + str(acc) + "%\n\n")
        my_file.write("levels = " + str(layers) + ";\n\n")
        my_file.write("n = [" + ', '.join(map(str,hidden)) + "];\n\n")
        
        # Write weights
        my_file.write("W = [" + "\n")

        for i in range(len(weights)):
            temp = weights[i]
            row, col = temp.shape
            for j in range(row):
                if (i+j == 0):
                    prefix = ""
                else:
                    prefix = sep
                my_file.write(prefix + sep.join(map(str, temp[j])) + "\n")

        my_file.write("];" + "\n\n")

        # Write bias
        my_file.write("B = [" + "\n")

        for i in range(len(bias)):
            temp = bias[i]
            row,  = temp.shape

            if (i == 0):
                prefix = ""
            else:
                prefix = sep

            my_file.write(prefix+ sep.join(map(str, temp)) + "\n")
        my_file.write("];" + "\n")


################################################################################
# Logs features in the files
################################################################################
def get_features(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    shift = 2
    activations = torch.zeros(len(val_loader.dataset),shift+1024)
    #print(len(val_loader.dataset))
    itr = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        if args.half:
            input_var = input_var.half()


        # output of feature boxes to contain the activations
        fc1_features = torch.zeros(input_var.shape[0],512)
        fc2_features = torch.zeros(input_var.shape[0],512)

        def copy_1(m, i, o):
            fc1_features.copy_(o.data)
        def copy_2(m, i, o):
            fc2_features.copy_(o.data)


        # attach hooks
        h1 = model.classifier[1].register_forward_hook(copy_1)
        h2 = model.classifier[4].register_forward_hook(copy_2)

        # compute output
        output,_ = model(input_var)

        # remove hooks
        h1.remove()
        h2.remove()

        start_id = i    *args.test_batch_size
        end_id   = (i+1)*args.test_batch_size

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        activations[start_id:end_id,0]     = target.squeeze()
        activations[start_id:end_id,1]     = pred.squeeze()
        activations[start_id:end_id,(shift+0)  :(shift+512) ]  = F.relu(fc1_features).reshape(input.shape[0],-1)
        activations[start_id:end_id,(shift+512):(shift+1024)]  = F.relu(fc2_features).reshape(input.shape[0],-1)

        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1,_,_ = accuracy(output.data, target)
        prec1 = prec1[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return activations, top1.avg

##########################################################################
# get a binary mask for stably inactive 
##########################################################################
def get_mask(out_before_relu):
    mask = (out_before_relu > 0).sum(dim=0) > 0
    return mask

################################################################################
# Logs features in the files
################################################################################
def eval_active_state(val_loader, model, criterion, fcnn_flag):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    N    = len(val_loader.dataset)

    if not fcnn_flag:
        LBLK = 2 # how many layers in a linear block 
        CBLK = 3 # how many layers in a conv block
        conv_cnt = model.conv_features.module.__len__() // CBLK
        conv_out_size = [[6,28,28], [16,10,10]]
        conv_active_states = [
                torch.zeros([N, 6,28,28]), torch.zeros([N, 16,10,10])]
    else:
        LBLK = 3 
        conv_cnt = 0
        conv_active_states = []
    linear_cnt = model.features.module.__len__() // LBLK
    linear_width = [model.features.module[i*LBLK].out_features for i in range(linear_cnt)]
    active_states = [torch.zeros(N, linear_width[i]) for i in range(linear_cnt)]
    batch_size = val_loader.batch_size
        
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for ii, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        if args.half:
            input_var = input_var.half()
        
        if fcnn_flag:
            input_var = input_var.view(input_var.shape[0],-1)

        linear = {}
        def get_linear(name):
            def hook(m, i, o):
                linear[name] = o.detach().cpu()
            return hook
        
        conv = {}
        def get_conv(name):
            def hook(m, i, o):
                conv[name] = o.detach().cpu()
            return hook
        # attach hooks
        hooks = []
        for layer_i in range(linear_cnt):
            hooks.append(model.features.module[layer_i*LBLK].register_forward_hook(get_linear(f'fc{layer_i+1}')))

        conv_hooks = []
        for layer_i in range(conv_cnt):
            conv_hooks.append(model.conv_features.module[layer_i*CBLK].register_forward_hook(get_conv(f'conv{layer_i+1}')))

        # compute output
        output,_ = model(input_var)

        # remove hooks
        for layer_i in range(linear_cnt):
            hooks[layer_i].remove()
        for layer_i in range(conv_cnt):
            conv_hooks[layer_i].remove()
            

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        start_id = ii        * batch_size
        end_id   = start_id + input_var.shape[0]
        for layer_i in range(linear_cnt):
            active_states[layer_i][start_id:end_id] = linear[f'fc{layer_i+1}'] >= 0
        for layer_i in range(conv_cnt):
            conv_active_states[layer_i][start_id:end_id] = conv[f'conv{layer_i+1}'] >= 0
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1,_,_ = accuracy(output.data, target_var)
        prec1 = prec1[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f' * Prec@1 {top1.avg:.3f}  Elapsed time: {batch_time.sum:.3f}s')
    all_states = []
    all_states.extend([s.reshape(s.shape[0],-1) for s in conv_active_states]) # flatten the states
    all_states.extend(active_states)

    masks = []
    masks.extend([get_mask(s) for s in conv_active_states])
    masks.extend([get_mask(s) for s in active_states])
    return all_states, top1.avg, masks



################################################################################
# Get the stable neuron indices from the active states
################################################################################
def find_stable_neurons(active_states):
    stably_active_neurons = []
    stably_inactive_neurons = []
    for layer_idx,layer_state in enumerate(active_states):
        # width is the neuron number in this layer
        data_size, width = layer_state.shape

        #import pdb;pdb.set_trace()
        # get the index of stable neurons in this layer 
        stably_active_idx = (layer_state.sum(dim=0) == data_size).nonzero(as_tuple=False)[:,None] 
        stably_inactive_idx = (layer_state.sum(dim=0) == 0).nonzero(as_tuple=False)[:,None]
        
        # concatenate the layer index
        stably_active_idx = torch.cat(
                [torch.ones_like(stably_active_idx) * (layer_idx+1), stably_active_idx], dim=1)
        stably_inactive_idx = torch.cat(
                [torch.ones_like(stably_inactive_idx) * (layer_idx+1), stably_inactive_idx], dim=1)
      
        print(f'{layer_idx}-th layer: stably_active={stably_active_idx.shape[0]}, stably_inactive={stably_inactive_idx.shape[0]}')
        stably_active_neurons.append(stably_active_idx) 
        stably_inactive_neurons.append(stably_inactive_idx) 
    
    stably_active_neurons = torch.cat(stably_active_neurons, dim=0)
    stably_inactive_neurons = torch.cat(stably_inactive_neurons, dim=0)
    
    print(f'Overall stably active: {stably_active_neurons.shape[0]}, stably inactive: {stably_inactive_neurons.shape[0]}')

    return stably_active_neurons, stably_inactive_neurons

################################################################################
# Writing tensor to csv files
################################################################################
def write_tensor_to_csv_file(subtensor,file_path):
    np.savetxt(file_path, np.asarray(subtensor), delimiter=" ")



################################################################################
# Get the prior and prefixed activations
################################################################################
def get_prior_activations(labels, activations_file_path):
    activations = np.loadtxt(activations_file_path,  delimiter=',')

    mask = np.zeros(activations.shape)
    mask[activations == 1.0] = 1
    mask[activations == 0.0] = 1

    # get the tensors now
    activations = torch.from_numpy(activations)
    mask = torch.from_numpy(mask)

    output1 = activations[labels,:]
    output2 = mask[labels,:]

    return output1, output2



################################################################################
# Original FGSM attack code
################################################################################
def basic_fgsm_attack(image, step_size, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image
    perturbed_image = image + step_size*sign_data_grad

    return perturbed_image


################################################################################
# FGSM attack code
################################################################################
def fgsm_attack(image, step_size, data_grad):
    perturbed_image = basic_fgsm_attack(image, step_size, data_grad)

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image



################################################################################
# PGD attack code
# Documentation from https://foolbox.readthedocs.io/en/latest/modules/attacks/gradient.html#foolbox.attacks.LinfinityBasicIterativeAttack
# Original paper https://arxiv.org/pdf/1607.02533.pdf
################################################################################
def pgd_attack(image, epsilon, data_grad, step_size, n_iterations):

    perturbed_image = image.clone()
    zero_tensor     = torch.zeros(image.shape).type(image.type())
    ones_tensor     = torch.ones (image.shape).type(image.type())

    for i in range(n_iterations):
        perturbed_image_new = basic_fgsm_attack(perturbed_image, step_size, data_grad)

        # Adding clipping to remain in epsilon neighbourhood of the original image
        temp = torch.max(perturbed_image_new, image - epsilon)
        temp = torch.max(temp, zero_tensor)

        temp = torch.min(temp, image + epsilon)
        perturbed_image = torch.min(temp, ones_tensor)

    # Return the perturbed image
    return perturbed_image



################################################################################
# Test adversarial attacks on the model
################################################################################
def test_adversarial(model, criterion, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # switch to evaluate mode
    model.eval()

    for i, (data, target) in enumerate(test_loader):

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        if args.half:
            data = data.half()

        # compute output
        output,_ = model(data)

        init_pred = (output.max(1, keepdim=True)[1]) # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        output = output.float()
        loss = loss.float()

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output,_ = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def get_net_width(model_path):
    ckp_path = os.path.join(model_path, 'pruned_checkpoint_120.tar')
    if not os.path.exists(ckp_path):
        print(f'No pruned model for {ckp_path}')
    ckp = torch.load(ckp_path) 
    w_names = sorted([name for name in ckp['state_dict'].keys()
                            if 'weight' in name and 'features' in name])
    widths = []
    for name in w_names:
        widths.append(ckp['state_dict'][name].shape[0])
    return widths

################################################################################
# Sets the learning rate to the initial LR decayed by 2 every 30 epochs.
# Adjunct. No longer needed
################################################################################
def adjust_learning_rate(optimizer, epoch):
    if epoch < 100:
        lr = 0.01
    elif epoch < 150:
        lr = 0.005
    #lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
