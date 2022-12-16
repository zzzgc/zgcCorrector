# -*- coding: utf-8 -*-
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertForMaskedLM, ElectraTokenizerFast
import argparse
from collections import OrderedDict
from loguru import logger
sys.path.append('../..')

from reader import make_loaders, DataCollator
from defaults import _C as cfg
from base_model import Model as DemoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    torch.multiprocessing.set_start_method('spawn', force=True)
    cfg = args_parse()

    # 如果不存在训练文件则先处理数据
    logger.info(f'load model, model arch: {cfg.MODEL.NAME}')
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
    
    # discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    d_tokenizer = ElectraTokenizerFast.from_pretrained(cfg.MODEL.DETECT_MODEL)

    # 加载数据
    train_loader, valid13_loader, valid14_loader, valid15_loader = make_loaders(
                                                           train_path=cfg.DATASETS.TRAIN,
                                                           valid13_path=cfg.DATASETS.VALID13, 
                                                           valid14_path=cfg.DATASETS.VALID14, 
                                                           valid15_path=cfg.DATASETS.VALID15,
                                                           batch_size=cfg.SOLVER.BATCH_SIZE, 
                                                           num_workers=4, 
                                                           tk=tokenizer,
                                                           d_tk=d_tokenizer)

    model = DemoModel(
        cfg,
        tokenizer,
        valid13_loader,
        valid14_loader,
        valid15_loader,
        cfg.MODEL.BERT_CKPT,
        cfg.MODEL.BERT_CKPT,
        cfg.MODEL.BERT_CKPT,
        cfg.MODEL.DETECT_MODEL
        )

    # 加载之前保存的模型，继续训练
    if cfg.MODEL.WEIGHTS and os.path.exists(cfg.MODEL.WEIGHTS):
        model.load_from_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS, cfg=cfg, map_location=device, tokenizer=tokenizer)
    # 配置模型保存参数
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.OUTPUT_DIR,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    # 训练模型
    logger.info('train model ...')
    trainer = pl.Trainer(max_epochs=cfg.SOLVER.MAX_EPOCHS,
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
    logger.info(f'ckpt_path: {ckpt_path}')
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        # 先保存原始transformer bert model
        tokenizer.save_pretrained(cfg.OUTPUT_DIR)
        bert = BertForMaskedLM.from_pretrained(cfg.MODEL.BERT_CKPT)
        bert.save_pretrained(cfg.OUTPUT_DIR)
        state_dict = torch.load(ckpt_path)['state_dict']
        new_state_dict = OrderedDict()
        if cfg.MODEL.NAME in ['macbert4csc']:
            for k, v in state_dict.items():
                if k.startswith('bert.'):
                    new_state_dict[k[5:]] = v
        else:
            new_state_dict = state_dict
        # 再保存finetune训练后的模型文件，替换原始的pytorch_model.bin
        torch.save(new_state_dict, os.path.join(cfg.OUTPUT_DIR, 'pytorch_model.bin'))
    # 进行测试的逻辑同训练


if __name__ == '__main__':
    main()
