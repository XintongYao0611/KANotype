from kanotype import config
from kanotype import train
from kanotype import predict
import os
import time
import logging
start_time = time.time()

conf = config.ConfObj(
    ref_ad='./data/hPancreas_train_adata.h5ad',
    query_ad='./data/hPancreas_test_adata.h5ad',    
    gmt_file='GO_bp.gmt',
    label_name='Celltype2',
    lr=1e-3,
    n_epoch=10,
    use_kan=True,
    batch_size=8,
    unknown_threshold=0,
    result_dir='./results/HumanPancreas_demo'
)

run_train = True
if run_train:
    train.run_training(conf)

predict.run_prediction(conf, conf.best_epoch)
end_time = time.time()
time_take = end_time - start_time

logging.basicConfig(filename=os.path.join(conf.result_dir, 'time_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("=============== Human Pancreas ===============")
logging.info(f"running time: {time_take} s")
logging.info(f"best epoch: {conf.best_epoch} s")
logging.info("=============== Human Pancreas ===============\n\n\n")

