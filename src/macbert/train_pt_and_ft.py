# -*- coding: utf-8 -*-
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertForMaskedLM
import argparse
from collections import OrderedDict
from loguru import logger
sys.path.append('../..')

from reader import make_loaders, DataCollator
from macbert4csc import MacBert4Csc
from defaults import _C as cfg
from base_model import DemoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def args_parse(config_file=''):
    parser = argparse.ArgumentParser(description="csc")
    parser.add_argument(
        "--config_file", default="train_macbert4csc.yml", help="path to config file", type=str
    )
    parser.add_argument("--opts", help="Modify config options using the command-line key value", default=[],
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    config_file = args.config_file or config_file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger.info(args)

    if config_file != '':
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def main():
    torch.multiprocessing.set_start_method('spawn')
    cfg = args_parse()

    # 如果不存在训练文件则先处理数据
    logger.info(f'load model, model arch: {cfg.MODEL.NAME}')
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    # 加载数据
    train_loader, valid13_loader, valid14_loader, valid15_loader, train_13_14_15_loader = make_loaders(train_path=cfg.DATASETS.TRAIN,
                                                           valid13_path=cfg.DATASETS.VALID13, valid14_path=cfg.DATASETS.VALID14, valid15_path=cfg.DATASETS.VALID15,
                                                           batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=4, tk=tokenizer, train_13_14_15=cfg.DATASETS.TRAIN13_14_15)

    for epoch in range(30):
        model = DemoModel(cfg, tokenizer, valid13_loader, valid14_loader, valid15_loader)
        if epoch != 0:
            model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        # 加载之前保存的模型，继续训练

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        ckpt_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.OUTPUT_DIR,
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        # 训练模型
        logger.info('27w pre-train model ...')
        trainer = pl.Trainer(max_epochs=1,
                            gpus=None if device == torch.device('cpu') else cfg.MODEL.GPU_IDS,
                            accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD_BATCHES,
                            val_check_interval=cfg.MODEL.STEP[0],
                            callbacks=[ckpt_callback])
        # 进行训练
        # train_loader中有数据
        torch.autograd.set_detect_anomaly(True)
        trainer.fit(model, train_loader, valid13_loader)
        logger.info('train model done.')

        # 模型转为transformers可加载
        if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
            ckpt_path = ckpt_callback.best_model_path
        elif cfg.MODEL.WEIGHTS and os.path.exists(cfg.MODEL.WEIGHTS):
            ckpt_path = cfg.MODEL.WEIGHTS
        else:
            ckpt_path = ''
        
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        
        logger.info(ckpt_path)
        logger.info('13_14_15 fine tuning model ...')
        ckpt_callback_ft = ModelCheckpoint(
            monitor='val_loss',
            dirpath=cfg.OUTPUT_DIR,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=1,
            mode='min'
        )
        trainer = pl.Trainer(max_epochs=30,
                            gpus=None if device == torch.device('cpu') else cfg.MODEL.GPU_IDS,
                            accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD_BATCHES,
                            val_check_interval=cfg.MODEL.STEP[0],
                            callbacks=[ckpt_callback_ft])
        # 进行训练
        # train_loader中有数据
        torch.autograd.set_detect_anomaly(True)
        trainer.fit(model, train_13_14_15_loader, valid13_loader)
        logger.info('the epoch ' + str(epoch) + ' model done.')
        
        

if __name__ == '__main__':
    main()
