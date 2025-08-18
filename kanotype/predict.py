import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
import time
import logging
from .config import ConfObj
from .utils import *
from .models import Transformer


def predict_adata(conf:ConfObj, model:Transformer, pathway_name:str=None, gene_name:str=None):       
    start_time = time.time()
    model.eval()
    parm={}

    if not conf.dic:
        from .dataset import splitDataSet
        _, _, _, _, inverse, _ = splitDataSet(conf.ref_ad, conf.label_name)
        dictionary = pd.DataFrame(inverse,columns=[conf.label_name])
        dictionary.loc[(dictionary.shape[0])] = 'unknown'
        dic = {}
        for i in range(len(dictionary)):
            dic[i] = dictionary[conf.label_name][i]
        conf.dic = dic

    for name,parameters in model.named_parameters():
        parm[name]=parameters.detach().cpu().numpy()
    
    latent = torch.empty([0,conf.embed_dim]).cpu()
    att = torch.empty([0,(len(conf.pathway))]).cpu()
    if pathway_name and gene_name:
        pathway_index = np.where(conf.pathway == pathway_name)[0][0]
        gene_index = conf.genes.index(gene_name)
        kan_embed = torch.empty([0, len(conf.pathway), conf.embed_dim]).cpu()
    
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)      
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class, pre_class]
    att = np.c_[att, predict_class, pre_class]
    adata_list = []

    n_line=0
    n_step=10000
    all_line = conf.query_ad.shape[0]
    while (n_line) <= all_line:
        if (all_line-n_line)%conf.batch_size != 1:
            expdata = pd.DataFrame(todense(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(conf.query_ad.var_names).tolist())
            n_line = n_line+n_step
        else:
            expdata = pd.DataFrame(todense(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(conf.query_ad.var_names).tolist())
            n_line = (all_line-n_line-2)

        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                  batch_size=conf.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                exp = data
                lat, pre, weights, kan_base_weight, kan_spline_weight = model(exp.to(conf.device))
                if pathway_name and gene_name:
                    exp_mask = torch.zeros_like(exp)  
                    exp_mask[:, gene_index] = 1
                    masked_exp = exp * exp_mask
                    kan_embed, _, _ = model.feature_embed(masked_exp.to(conf.device))
                    kan_embed = kan_embed[:, pathway_index, :]
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre,1)
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0) 
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= conf.unknown_threshold: 
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        predict_class = np.r_[predict_class, conf.num_classes]
                    pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
                l_p = torch.squeeze(lat).cpu().numpy()
                att = torch.squeeze(weights).cpu().numpy()
                pre = pre.numpy()
                if pathway_name and gene_name:
                    kan_emb = torch.squeeze(kan_embed).cpu().numpy()
                    kan_emb = kan_emb.astype('float32')
                meta = np.c_[predict_class,pre_class]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction','Probability']
                meta.index = meta.index.astype('str')

                att = att[:,0:(len(conf.pathway)-conf.n_unannotated)]
                att = att.astype('float32')
                l_p = l_p.astype('float32')
                pre = pre.astype('float32')
                varinfo = pd.DataFrame(pd.DataFrame(conf.pathway).iloc[0:len(conf.pathway)-conf.n_unannotated,0].values,index=pd.DataFrame(conf.pathway).iloc[0:len(conf.pathway)-conf.n_unannotated,0],columns=['pathway_index'])
                new = sc.AnnData(att, obs=meta, var = varinfo)
                new.obsm['latent'] = l_p
                new.obsm['softmax'] = pre
                if pathway_name and gene_name:
                    new.obsm['kan_embedding'] = kan_emb
                new.var.index.name = str(new.var.index.name)
                adata_list.append(new)
    print("Prediction finished!")

    new = ad.concat(adata_list)
    new.obs.index = conf.query_ad.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(conf.dic)
    new.obs[conf.query_ad.obs.columns] = conf.query_ad.obs[conf.query_ad.obs.columns].values
    end_time = time.time()
    test_time = end_time - start_time
    logging.info(f"test time consumption: {test_time} s")
    return(new)


def get_kan_embed(conf:ConfObj, model, pathway_name:str, gene_name:str):

    start_time = time.time()
    model.eval()
    parm={}
    for name,parameters in model.named_parameters():
        parm[name]=parameters.detach().cpu().numpy()
    
    latent = torch.empty([0,conf.embed_dim]).cpu()
    att = torch.empty([0,(len(conf.pathway))]).cpu()
    predict_class = np.empty(shape=0)
    pre_class = np.empty(shape=0)      
    latent = torch.squeeze(latent).cpu().numpy()
    l_p = np.c_[latent, predict_class,pre_class]
    att = np.c_[att, predict_class,pre_class]
    adata_list = []

    n_line=0
    n_step=10000
    all_line = conf.query_ad.shape[0]
    while (n_line) <= all_line:
        if (all_line-n_line)%conf.batch_size != 1:
            expdata = pd.DataFrame(todense(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(conf.query_ad.var_names).tolist())
            n_line = n_line+n_step
        else:
            expdata = pd.DataFrame(todense(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line-2))]),index=np.array(conf.query_ad[n_line:n_line+min(n_step,(all_line-n_line-2))].obs_names).tolist(), columns=np.array(conf.query_ad.var_names).tolist())
            n_line = (all_line-n_line-2)
        expdata = np.array(expdata)
        expdata = torch.from_numpy(expdata.astype(np.float32))
        data_loader = torch.utils.data.DataLoader(expdata,
                                                  batch_size=conf.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                exp = data
                lat, pre, weights, kan_base_weight, kan_spline_weight = model(exp.to(conf.device))
                pre = torch.squeeze(pre).cpu()
                pre = F.softmax(pre,1)
                predict_class = np.empty(shape=0)
                pre_class = np.empty(shape=0) 
                for i in range(len(pre)):
                    if torch.max(pre, dim=1)[0][i] >= conf.unknown_threshold: 
                        predict_class = np.r_[predict_class,torch.max(pre, dim=1)[1][i].numpy()]
                    else:
                        predict_class = np.r_[predict_class, conf.num_classes]
                    pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]     
                l_p = torch.squeeze(lat).cpu().numpy()
                att = torch.squeeze(weights).cpu().numpy()
                meta = np.c_[predict_class,pre_class]
                meta = pd.DataFrame(meta)
                meta.columns = ['Prediction','Probability']
                meta.index = meta.index.astype('str')

                att = att[:,0:(len(conf.pathway)-conf.n_unannotated)]
                att = att.astype('float32')
                varinfo = pd.DataFrame(pd.DataFrame(conf.pathway).iloc[0:len(conf.pathway)-conf.n_unannotated,0].values,index=pd.DataFrame(conf.pathway).iloc[0:len(conf.pathway)-conf.n_unannotated,0],columns=['pathway_index'])
                new = sc.AnnData(att, obs=meta, var = varinfo)
                new.var.index.name = str(new.var.index.name)
                adata_list.append(new)

    new = ad.concat(adata_list)
    new.obs.index = conf.query_ad.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(conf.dic)
    new.obs[conf.query_ad.obs.columns] = conf.query_ad.obs[conf.query_ad.obs.columns].values
    end_time = time.time()
    test_time = end_time - start_time
    logging.info(f"test time consumption: {test_time} s")
    return(new)


def run_prediction(conf:ConfObj, model_epo:int):
    model = Transformer(device=conf.device,
                        num_classes=conf.num_classes, 
                        num_genes=len(conf.genes), 
                        mask = conf.mask,
                        embed_dim=conf.embed_dim,
                        depth=conf.vit_dep,
                        num_heads=conf.vit_heads,
                        drop_ratio=conf.vit_drop, attn_drop_ratio=0.5, drop_path_ratio=0.5
                        ).to(conf.device)
    model_weight_path = os.path.join(conf.result_dir, 'models', f'model_{model_epo}.pth')
    model.load_state_dict(torch.load(model_weight_path, map_location=conf.device))

    new_adata = predict_adata(conf=conf, model=model)
    new_adata.obs = new_adata.obs.astype(str)
    new_adata.write(os.path.join(conf.result_dir, 'adata', f'adata_epo{model_epo}.h5ad'))
