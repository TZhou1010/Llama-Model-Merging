import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
#sys.path.append('/home/taskarithmetic/')
sys.path.append('/Volumes/Samsung_T3/AdaMergingData')
sys.path.append(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ['EuroSAT', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-16'
args = parse_arguments()
args.data_location = os.path.join("/Volumes", "Samsung_T3", "AdaMergingData","dataset")
args.model = model
args.save = '/Volumes/Samsung_T3/AdaMergingData/checkpoint/' + model
args.logs_path = os.path.join("logs", model)
pretrained_checkpoint = '/Volumes/Samsung_T3/AdaMergingData/checkpoint/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '/Volumes/Samsung_T3/AdaMergingData/checkpoint/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

task_vector_sum = sum(task_vectors)

scaling_coef_ = 0.3

image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

accs = []
for dataset in exam_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
