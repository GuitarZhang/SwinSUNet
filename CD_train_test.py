import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as tr
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from math import exp

PATCH_SIZE = 224

def kappa(tp, tn, fp, fn):
    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) / (N * N)
    return (p0 - pe) / (1 - pe)

def predict_one_pic(dset, net, save_path):
    net.eval()
    for img_index in range(len(dset.names)):
      if '18' in dset.names[img_index]:
        I1_full, I2_full, cm_full = dset.get_img(dset.names[img_index])
        img_size = cm_full.shape
        p_img = np.zeros(img_size)

        n1 = img_size[0] // PATCH_SIZE
        if img_size[0] % PATCH_SIZE > 0:
            n1 = n1 + 1
        n2 = img_size[1] // PATCH_SIZE
        if img_size[1] % PATCH_SIZE > 0:
            n2 = n2 + 1
        for i in range(n1):
            start_i = i * PATCH_SIZE
            end_i = min((i + 1) * PATCH_SIZE, img_size[0])
            for j in range(n2):
                start_j = j * PATCH_SIZE
                end_j = min((j + 1) * PATCH_SIZE, img_size[1])

                I1 = I1_full[:, start_i:end_i, start_j:end_j]
                I2 = I2_full[:, start_i:end_i, start_j:end_j]
                cm = cm_full[start_i:end_i, start_j:end_j]

                ori_size = cm.shape

                if cm.shape[0] < PATCH_SIZE or cm.shape[1] < PATCH_SIZE:
                    I1 = I1_full[:, end_i - PATCH_SIZE:end_i, end_j - PATCH_SIZE:end_j]
                    I2 = I2_full[:, end_i - PATCH_SIZE:end_i, end_j - PATCH_SIZE:end_j]

                I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
                I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
                
                outputs = net(I1, I2)

                if cm.shape[0] < PATCH_SIZE or cm.shape[1] < PATCH_SIZE:
                    outputs = outputs[:,:,-cm.shape[0]:,-cm.shape[1]:]

                _, predicted = torch.max(outputs.data, 1)
                predicted  = predicted.view(cm.shape[0],cm.shape[1])*255
                predicted  = predicted.cpu().numpy()
                
                p_img[start_i:end_i, start_j:end_j] = predicted

        # io.imshow(p_img)
        plt.show()
        io.imsave(save_path + 'predicted_' +  dset.names[img_index] + '.jpg', p_img)
        # io.imsave(save_path + 'label_' + dset.names[img_index]+'.jpg', cm_full)

def test_batch(loader, net, test_loader, net_name):
    tot_count = 0

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    net.eval()
    for batch, cootds in test_loader:
        I1 = Variable(batch['I1'].float().cuda())
        I2 = Variable(batch['I2'].float().cuda())
        labels = Variable(batch['label'].cuda())
        outputs = net(I1, I2)

        cootds = cootds[1]

        for i in range(I1.shape[0]):
            output = outputs[i:i+1]
            label = labels[i:i+1]

            h = cootds[1][i]- cootds[0][i]
            w = cootds[3][i] - cootds[2][i]

            h_s = output.shape[2]
            w_s = output.shape[3]
            output = output[:,:,h_s - h:, w_s - w:]
            label = label[:,h_s - h:, w_s - w:]

            p_n = np.prod(label.shape)
                  
            tot_count += p_n
            
            _, predicted = torch.max(output.data, 1)

            c = (predicted.int() == label.data.int())

            pr = (predicted.int() > 0).cpu().numpy()
            gt = (label.data.int() > 0).cpu().numpy()

            a = np.zeros(2)
            a[1] = np.logical_and(pr, gt).sum()
            a[0] = np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()

            class_correct[0] += a[0]
            class_correct[1] += a[1]

            a[1] = gt.sum()
            a[0] = c.size(1) * c.size(2) - a[1]

            class_total[0] += a[0]
            class_total[1] += a[1]
            tp += np.logical_and(pr, gt).sum()
            tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
            fp += np.logical_and(pr, np.logical_not(gt)).sum()
            fn += np.logical_and(np.logical_not(pr), gt).sum()
    print(tp, tn, fp, fn, tp + tn + fp + fn)
    
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)
        class_accuracy[i] =  float(class_accuracy[i])

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    dice = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)
    f1_score = (2 * prec * rec) / (prec + rec)
    
    pr_rec = [prec, rec, dice, prec_nc, rec_nc]
    
    k = kappa(tp, tn, fp, fn)
    return {'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'OA': net_accuracy, 
            'precision': prec, 
            'recall': rec, 
            'f1_score': f1_score
            }

def get_epoch_prediction(epoch_list, epoch_index, dataset, net, criterion, net_name):
    epoch_loss, epoch_accuracy, epoch_nochange_accuracy, epoch_change_accuracy, epoch_precision, epoch_recall, epoch_Fmeasure = epoch_list
    val_interval = 2 #1or2
    if val_interval == 2 and epoch_index % val_interval == 1:
        epoch_loss[epoch_index], epoch_accuracy[epoch_index], cl_acc, pr_rec = test(dataset, net, criterion, net_name)
        epoch_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_change_accuracy[epoch_index] = cl_acc[1]
        epoch_precision[epoch_index] = pr_rec[0]
        epoch_recall[epoch_index] = pr_rec[1]
        epoch_Fmeasure[epoch_index] = pr_rec[2]

        if epoch_index !=1:
            epoch_loss[epoch_index - 1]  = epoch_loss[epoch_index]/2 + epoch_loss[epoch_index-2]/2
            epoch_accuracy[epoch_index - 1] = epoch_accuracy[epoch_index]/2 + epoch_accuracy[epoch_index-2]/2
            epoch_nochange_accuracy[epoch_index - 1] = (epoch_nochange_accuracy[epoch_index] + epoch_nochange_accuracy[epoch_index - 2]) /2
            epoch_change_accuracy[epoch_index - 1] = (epoch_change_accuracy[epoch_index] + epoch_change_accuracy[epoch_index - 2]) /2
            epoch_precision[epoch_index - 1] = (epoch_precision[epoch_index] + epoch_precision[epoch_index - 2]) /2
            epoch_recall[epoch_index - 1] = (epoch_recall[epoch_index] + epoch_recall[epoch_index - 2]) /2
            epoch_Fmeasure[epoch_index - 1] = (epoch_Fmeasure[epoch_index] + epoch_Fmeasure[epoch_index - 2]) /2
        if epoch_index ==1:
            epoch_loss[epoch_index - 1]  = epoch_loss[epoch_index]/2
            epoch_accuracy[epoch_index - 1] = epoch_accuracy[epoch_index]/2
            epoch_nochange_accuracy[epoch_index - 1] = (epoch_nochange_accuracy[epoch_index] ) /2
            epoch_change_accuracy[epoch_index - 1] = (epoch_change_accuracy[epoch_index] ) /2
            epoch_precision[epoch_index - 1] = (epoch_precision[epoch_index]) /2
            epoch_recall[epoch_index - 1] = (epoch_recall[epoch_index]) /2
            epoch_Fmeasure[epoch_index - 1] = (epoch_Fmeasure[epoch_index]) /2
    elif val_interval == 1:
        epoch_loss[epoch_index], epoch_accuracy[epoch_index], cl_acc, pr_rec = test(dataset, net, criterion, net_name)
        epoch_nochange_accuracy[epoch_index] = cl_acc[0]
        epoch_change_accuracy[epoch_index] = cl_acc[1]
        epoch_precision[epoch_index] = pr_rec[0]
        epoch_recall[epoch_index] = pr_rec[1]
        epoch_Fmeasure[epoch_index] = pr_rec[2]
    return epoch_loss, epoch_accuracy, epoch_nochange_accuracy, epoch_change_accuracy, epoch_precision, epoch_recall, epoch_Fmeasure

def test(dset, net, criterion, net_name):
    net.eval()
    tot_loss = 0
    tot_count = 0
    tot_accurate = 0
    
    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for img_index in range(len(dset.names)):
        if img_index % 50 == 0:
            print(img_index, '/', len(dset.names))
        I1_full, I2_full, cm_full = dset.get_img(dset.names[img_index])
        
        img_size = cm_full.shape

        n1 = img_size[0] // PATCH_SIZE
        n2 = img_size[1] // PATCH_SIZE

        for i in range(n1):
            start_i = i * PATCH_SIZE
            end_i = min((i + 1) * PATCH_SIZE, img_size[0])
            for j in range(n2):
                start_j = j * PATCH_SIZE
                end_j = min((j + 1) * PATCH_SIZE, img_size[1])

                I1 = I1_full[:, start_i:end_i, start_j:end_j]
                I2 = I2_full[:, start_i:end_i, start_j:end_j]
                cm = cm_full[start_i:end_i, start_j:end_j]
        
                I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
                I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).float()).cuda()
                
                output = net(I1, I2)
                
                loss = criterion(output, cm.long())
                
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                        
                pr = (predicted.int() > 0).cpu().numpy()
                gt = (cm.data.int() > 0).cpu().numpy()

                a = np.zeros(2)
                a[1] = np.logical_and(pr, gt).sum()
                a[0] = np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()

                class_correct[0] += a[0]
                class_correct[1] += a[1]

                a[1] = gt.sum()
                a[0] = c.size(1) * c.size(2) - a[1]

                class_total[0] += a[0]
                class_total[1] += a[1]
                
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()
    
    net_loss = tot_loss/tot_count
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)
    
    pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]
        
    return net_loss, net_accuracy, class_accuracy, pr_rec

def train(optimizer, scheduler,save_path, net, train_loader, criterion, train_dataset, test_dataset, net_name, n_epochs, save = True):
    best_f1_net_name = '.path.tar'
    t = np.linspace(1, n_epochs, n_epochs)
    
    epoch_test_loss = 0 * t
    epoch_test_accuracy = 0 * t
    epoch_test_change_accuracy = 0 * t
    epoch_test_nochange_accuracy = 0 * t
    epoch_test_precision = 0 * t
    epoch_test_recall = 0 * t
    epoch_test_Fmeasure = 0 * t
    fm = 0
    best_fm = 0
    
    plt.figure(num=1)
    plt.figure(num=2)
    plt.figure(num=3)
    
    for epoch_index in tqdm(range(n_epochs)):
        net.train()
        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(n_epochs))

        # tot_count = 0
        # tot_loss = 0
        # tot_accurate = 0
        # class_correct = list(0. for i in range(2))
        # class_total = list(0. for i in range(2))

        i_b = 0
        for batch in train_loader:
            I1 = Variable(batch['I1'].float().cuda())
            I2 = Variable(batch['I2'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))

            optimizer.zero_grad()
            output = net(I1, I2)
            if len(label.shape) == 2:
                label = label.view(1, label.shape[0], label.shape[1])
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            i_b = i_b + 1
            if i_b % 100 == 0:
                print(i_b, loss.data)
        scheduler.step()

        epoch_list = [epoch_test_loss, epoch_test_accuracy, epoch_test_nochange_accuracy, epoch_test_change_accuracy, \
            epoch_test_precision, epoch_test_recall, epoch_test_Fmeasure]
        epoch_test_loss, epoch_test_accuracy, epoch_test_nochange_accuracy, epoch_test_change_accuracy, \
            epoch_test_precision, epoch_test_recall, epoch_test_Fmeasure =\
            get_epoch_prediction(epoch_list, epoch_index, test_dataset, net, criterion, net_name)

        plt.figure(num=1)
        plt.clf()
        l1_2, = plt.plot(t[:epoch_index + 1], epoch_test_loss[:epoch_index + 1], label='Test loss')
        plt.legend(handles=[l1_2])
        plt.grid()
        plt.gcf().gca().set_xlim(left = 0)
        plt.title('Loss')

        plt.figure(num=2)
        plt.clf()
        l2_2, = plt.plot(t[:epoch_index + 1], epoch_test_accuracy[:epoch_index + 1], label='Test accuracy')
        plt.legend(handles=[l2_2])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        plt.title('Accuracy')

        plt.figure(num=3)
        plt.clf()
        l3_3, = plt.plot(t[:epoch_index + 1], epoch_test_nochange_accuracy[:epoch_index + 1], label='Test accuracy: no change')
        l3_4, = plt.plot(t[:epoch_index + 1], epoch_test_change_accuracy[:epoch_index + 1], label='Test accuracy: change')
        plt.legend(handles=[l3_3, l3_4])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 100)
        plt.title('Accuracy per class')

        plt.figure(num=4)
        plt.clf()
        l4_4, = plt.plot(t[:epoch_index + 1], epoch_test_precision[:epoch_index + 1], label='Test precision')
        l4_5, = plt.plot(t[:epoch_index + 1], epoch_test_recall[:epoch_index + 1], label='Test recall')
        l4_6, = plt.plot(t[:epoch_index + 1], epoch_test_Fmeasure[:epoch_index + 1], label='Test Dice/F1')
        plt.legend(handles=[l4_4, l4_5, l4_6])
        plt.grid()
        plt.gcf().gca().set_ylim(0, 1)
        plt.title('Precision, Recall and F-measure')
        
        fm = epoch_test_Fmeasure[epoch_index]
        if fm > best_fm:
            best_fm = fm
            save_str = save_path + 'net-best_epoch-' + str(epoch_index + 1) + '_fm-' + str(fm) + '.pth.tar'
            torch.save(net.state_dict(), save_str)
            best_f1_net_name = save_str

        all_save = 0
        if all_save:
            save_str = save_path + 'net_epoch-' + str(epoch_index + 1)  + '.pth.tar'
            torch.save(net.state_dict(), save_str)

        if save:
            im_format = 'png'

            plt.figure(num=1)
            plt.savefig(save_path+net_name + '-01-loss.' + im_format)

            plt.figure(num=2)
            plt.savefig(save_path+net_name + '-02-accuracy.' + im_format)

            plt.figure(num=3)
            plt.savefig(save_path+net_name + '-03-accuracy-per-class.' + im_format)

            plt.figure(num=4)
            plt.savefig(save_path+net_name + '-04-prec-rec-fmeas.' + im_format)
        
    out = {'test_loss': epoch_test_loss[-1],
           'test_accuracy': epoch_test_accuracy[-1],
           'test_nochange_accuracy': epoch_test_nochange_accuracy[-1],
           'test_change_accuracy': epoch_test_change_accuracy[-1]}
    
    return out, best_f1_net_name
