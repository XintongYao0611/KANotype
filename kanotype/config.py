import torch
import numpy as np
import os
import logging

from .utils import *
from .dataset import *


class ConfObj():
    def __init__(self,
                 ref_ad,
                query_ad,
                gmt_file,
                label_name,
                n_epoch,
                seed=1,
                use_kan=True,
                min_g=0,
                max_g=300,
                max_gs=300,
                n_unannotated=1,
                lr=1e-3,
                lrf=0.01,
                batch_size=8,
                vit_dep=2,
                vit_heads=4,
                vit_drop=0.5,
                embed_dim=48,
                early_stop=True,
                unknown_threshold=0.1,
                result_dir='result',
                top_n=5,
                max_epoch=100,
                use_last=False,
            ):
        self.seed=seed
        set_seed_all(seed=self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.ref_ad=get_ann_obj(ref_ad)
        self.query_ad=get_ann_obj(query_ad)

        self.n_epoch=n_epoch
        self.genes=list(self.ref_ad.var_names)
        self.input_dim=len(self.genes)
        self.label_name=label_name
        
        gmt_file = os.path.join(os.path.dirname(__file__), 'gmt', gmt_file)
        reactome_dict = read_gmt(gmt_file, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=self.genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        self.pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        self.mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        self.rep_mask = torch.tensor(np.repeat(self.mask,embed_dim,axis=1), dtype=torch.float).t()
        
        self.num_classes=get_n_types(conf_label_name=self.label_name, ann_obj=self.ref_ad)
        self.n_unannotated=n_unannotated
        self.dic = {}

        self.celltypes=list(set(self.ref_ad.obs[self.label_name].values))
        self.use_kan=use_kan
        self.lr=lr
        self.lrf=lrf
        self.embed_dim=embed_dim
        self.unknown_threshold=unknown_threshold
        self.batch_size=batch_size
        self.vit_dep=vit_dep
        self.vit_heads=vit_heads
        self.vit_drop=vit_drop

        self.gmt_file=gmt_file
        self.min_g=min_g
        self.max_g=max_g
        self.n_patches=len(self.pathway)
        self.top_n = top_n
        self.max_epoch=max_epoch
        self.use_last=use_last        
        self.result_dir = result_dir
        self.create_proj_dir()

        self.early_stop = early_stop
        self.best_epoch = n_epoch - 1

        # detection only
        self.best_model_pth = os.path.join(self.result_dir, 'models', f"best_model.pth")
        self.last_model_pth = os.path.join(self.result_dir, 'models', f"last_model.pth")

        print('Mask loaded!')

        logging.basicConfig(filename=os.path.join(self.result_dir, 'logs.txt'), level=logging.INFO,# 只记录 INFO 及以上的日志
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
        logging.info(f"training set: {ref_ad}\ntest set: {query_ad}\ngmt file: {gmt_file}")

    
    def create_proj_dir(self):
        mkdir_p(self.result_dir)
        mkdir_p(os.path.join(self.result_dir, 'models'))
        mkdir_p(os.path.join(self.result_dir, 'adata'))
        mkdir_p(os.path.join(self.result_dir, 'downstream'))
        mkdir_p(os.path.join(self.result_dir, 'weights'))
        mkdir_p(os.path.join(self.result_dir, 'weights/kan'))
        mkdir_p(os.path.join(self.result_dir, 'weights/attention'))
        mkdir_p(os.path.join(self.result_dir, 'accuracy'))
