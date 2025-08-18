import sys
import pandas as pd
from tqdm import tqdm
import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import logging
import gc

from .models import Transformer
from .config import ConfObj
from .utils import *
from .dataset import *


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_,_,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.4f}, acc: {accu_num.item() / sample_num:.4f}"
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_model(conf:ConfObj, model:Transformer, criterion, train_loader, val_loader):    
    pg = [p for p in model.parameters() if p.requires_grad]  
    optimizer = optim.SGD(pg, lr=conf.lr, momentum=0.9, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / conf.n_epoch)) / 2) * (1 - conf.lrf) + conf.lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    print("Starting Training Loop...")
    device = conf.device
    train_loss = []
    train_accu = []
    valid_loss = []
    valid_accu = []
    delta_valid_loss = [0.0, 0.0]
    for epoch in range(conf.n_epoch):
        epo_start_time = time.time()
        model.train()
        loss_function = criterion 
        accu_loss = torch.zeros(1).to(device) 
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()
        sample_num = 0
        data_loader = tqdm(train_loader)
        for step, data in enumerate(data_loader):
            exp, label = data
            sample_num += exp.shape[0]
            _,pred,_,_,_ = model(exp.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()
            loss = loss_function(pred, label.to(device))
            loss.backward()
            accu_loss += loss.detach()
            data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.4f}, acc: {accu_num.item() / sample_num:.4f}"
            
            optimizer.step() 
            optimizer.zero_grad()
            train_los = accu_loss.item() / (step + 1)
            train_acc = accu_num.item() / sample_num
            train_loss.append(train_los)
            train_accu.append(train_acc)

        scheduler.step() 
        val_los, val_acc = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)

        
        valid_loss.append(val_los)
        valid_accu.append(val_acc)

        if epoch == 0:
            threshold_valid_los = 0.05 if valid_loss[0] > 1.00 else 0.01
        else:
            delta_valid_loss[0] = delta_valid_loss[-1]
            delta_valid_loss[-1] = abs(valid_loss[epoch] - valid_loss[epoch - 1])
            if conf.early_stop and (epoch >= 2) and (delta_valid_loss[0] < threshold_valid_los) and (delta_valid_loss[-1] < threshold_valid_los):
                conf.best_epoch = epoch - 1
                logging.info(f"epoch {epoch}: train loss:{train_los}  train accu:{train_acc}")
                logging.info(f"epoch {epoch}: valid loss:{val_los}  valid accu:{val_acc}")
                logging.info(f"in epoch {epoch} reached early stop condition")
                torch.save(model.state_dict(), os.path.join(conf.result_dir, 'models', f'model_{epoch}.pth'))
                epo_end_time = time.time()
                epo_time = epo_end_time - epo_start_time
                logging.info(f"epoch {epoch}: training time consumption:{epo_time} s")
                print('Training finished!')
                break

        logging.info(f"epoch {epoch}: train loss:{train_los}  train accu:{train_acc}")
        logging.info(f"epoch {epoch}: valid loss:{val_los}  valid accu:{val_acc}")
        torch.save(model.state_dict(), os.path.join(conf.result_dir, 'models', f'model_{epoch}.pth'))
        epo_end_time = time.time()
        epo_time = epo_end_time - epo_start_time
        logging.info(f"epoch {epoch}: training time consumption:{epo_time} s")
    print('Training finished!')


def run_training(conf:ConfObj):
    exp_train, label_train, exp_valid, label_valid, inverse, genes = splitDataSet(conf.ref_ad, conf.label_name)
    dictionary = pd.DataFrame(inverse,columns=[conf.label_name])
    dictionary.loc[(dictionary.shape[0])] = 'unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[conf.label_name][i]
    conf.dic = dic
      
    train_dataset = MyDataSet(exp_train, label_train)
    valid_dataset = MyDataSet(exp_valid, label_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=conf.batch_size,
                                                shuffle=True,
                                                pin_memory=True,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=conf.batch_size,
                                                shuffle=False,
                                                pin_memory=True,drop_last=True)

    model = Transformer(device=conf.device,
                            num_classes=conf.num_classes, 
                            num_genes=len(conf.genes), 
                            mask = conf.mask,
                            embed_dim=conf.embed_dim,
                            depth=conf.vit_dep,
                            num_heads=conf.vit_heads,
                            drop_ratio=conf.vit_drop, attn_drop_ratio=0.5, drop_path_ratio=0.5
                            ).to(conf.device)

    train_model(conf=conf,
                    model=model,
                    criterion=torch.nn.CrossEntropyLoss(),
                    train_loader=train_loader,
                    val_loader=valid_loader)

    del model
    torch.cuda.empty_cache()
    gc.collect()
