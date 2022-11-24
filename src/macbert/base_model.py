# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import operator
from abc import ABC
from tkinter import SEL
from wsgiref.headers import tspecials
from loguru import logger
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
# from pycorrector.macbert import lr_scheduler
import lr_scheduler
from evaluate_util import compute_corrector_prf, compute_sentence_level_prf
from transformers import AutoModel, AutoTokenizer

class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
        "warmup_epochs": cfg.SOLVER.WARMUP_EPOCHS,
        "warmup_method": cfg.SOLVER.WARMUP_METHOD,

        # multi-step lr scheduler options
        "milestones": cfg.SOLVER.STEPS,
        "gamma": cfg.SOLVER.GAMMA,

        # cosine annealing lr scheduler options
        "max_iters": cfg.SOLVER.MAX_ITER,
        "delay_iters": cfg.SOLVER.DELAY_ITERS,
        "eta_min_lr": cfg.SOLVER.ETA_MIN_LR,

    }
    scheduler = getattr(lr_scheduler, cfg.SOLVER.SCHED)(**scheduler_args)
    return {'scheduler': scheduler, 'interval': cfg.SOLVER.INTERVAL}


class BaseTrainingEngine(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self)
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        return [optimizer], [scheduler]

    def on_validation_epoch_start(self) -> None:
        logger.info('Valid.')

    def on_test_epoch_start(self) -> None:
        logger.info('Testing...')


class CscTrainingModel(BaseTrainingEngine, ABC):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight
        self.w = cfg.MODEL.HYPER_PARAMS[0]

    def training_step(self, batch, batch_idx):
        # loss = torch.tensor(0, dtype=float)
        # loss.requires_grad_(True)  
        # return loss
        ori_text, cor_text, det_labels = batch['input_ids'], batch['labels'], batch['pos_labels']
        outputs = self.forward(batch)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(ori_text))
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch['input_ids'], batch['labels'], batch['pos_labels']
        outputs = self.forward(batch)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        cor_y = batch['labels']
        cor_y_hat *= batch['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label, l in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels, batch['len']):
            _src = src[1:l+1]
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
            results.append((_src, _tgt, _predict,))

        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def validation_epoch_end(self, outputs) -> None:
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
        loss = np.mean([out[0] for out in outputs])
        self.log('val_loss', loss)
        logger.info(f'loss: {loss}')
        logger.info(f'Detection:\n'
                    f'acc: {np.mean(det_acc_labels):.4f}')
        logger.info(f'Correction:\n'
                    f'acc: {np.mean(cor_acc_labels):.4f}')
        compute_corrector_prf(results, logger)
        compute_sentence_level_prf(results, logger)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        logger.info('Test.')
        self.validation_epoch_end(outputs)

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        return rst


class DemoModel(CscTrainingModel, ABC):
    def __init__(self, cfg, tokenizer, sighan13, sighan14, sighan15):
        super().__init__(cfg)
        self.cfg = cfg
        self.sighan13 = sighan13
        self.sighan14 = sighan14
        self.sighan15 = sighan15
        self.bert = AutoModel.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.correction = nn.Linear(self.bert.config.hidden_size, 21128)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer
        self.loss_fct = nn.CrossEntropyLoss()
        self.det_loss_fct = nn.BCELoss()
        
    def forward(self, batch):
        batch = {k:batch[k].to('cuda:0') for k in batch}
        out = self.bert(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask']
            )
        src_label = batch['labels']
        det_labels = batch['pos_labels']
        src_label[src_label == 0] = -100  # -100计算损失时会忽略
        det_labels[det_labels == 2] = 0  # -100计算损失时会忽略
        
        # 检错概率
        prob = self.detection(out.last_hidden_state)
        det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
        # pad部分不计算损失
        active_loss = batch['attention_mask'].view(-1, prob.shape[1]) == 1
        active_probs = prob.view(-1, prob.shape[1])[active_loss]
        active_labels = det_labels[active_loss]
        det_loss = det_loss_fct(active_probs, active_labels.float())
        # det_loss = self.det_loss_fct(self.sigmoid(prob).view(-1, prob.shape[1]).float(), det_labels.float())


        # 纠错loss
        out = self.correction(out.last_hidden_state)
        correct_loss = self.loss_fct(out.view(-1, self.bert.config.vocab_size), src_label.view(-1))
        # 检错loss，纠错loss，检错输出，纠错输出
        outputs = (det_loss,
                    correct_loss,
                    self.sigmoid(prob).squeeze(-1),
                    torch.argmax(out, dim=-1))
        batch = {k:batch[k].to('cpu') for k in batch}
        
        return outputs
    
    # def train_dataloader(self):
    #     return self.train_dataloader

    
    def validation_step(self, batch, batch_idx):
        self.log('val_loss', 0)
        return None

    def validation_epoch_end(self, outputs) -> None:
        logger.info('>>>> SIGHAN13')
        results = []
        for batch in self.sighan13:
            out = self.forward(batch)
            pre_c, pre_d = out[3].tolist(), out[2].tolist()
            src, trg, d_labels = batch['input_ids'].tolist(), batch['labels'].tolist(), batch['pos_labels'].tolist()
            l = batch['len'].tolist()
            for p_c, p_d, s, t, d_l, ll in zip(pre_c, pre_d, src, trg, d_labels, l):
                p_d = p_d[1:ll+1]
                s = s[1:ll+1]
                t = t[1:ll+1]
                p_c = p_c[1:ll+1]
                d_l = d_l[1:ll+1]
                results.append((s, p_c, t, p_d, d_l))
        # self.compute_corrector_prf(results)
        compute_corrector_prf(results, logger)
        compute_sentence_level_prf(results, logger)

        logger.info('>>>> SIGHAN14')
        results = []
        for batch in self.sighan14:
            out = self.forward(batch)
            pre_c, pre_d = out[3].tolist(), out[2].tolist()
            src, trg, d_labels = batch['input_ids'].tolist(), batch['labels'].tolist(), batch['pos_labels'].tolist()
            l = batch['len'].tolist()
            for p_c, p_d, s, t, d_l, ll in zip(pre_c, pre_d, src, trg, d_labels, l):
                p_d = p_d[1:ll+1]
                s = s[1:ll+1]
                t = t[1:ll+1]
                p_c = p_c[1:ll+1]
                d_l = d_l[1:ll+1]
                results.append((s, p_c, t, p_d, d_l))
        # self.compute_corrector_prf(results)
        compute_corrector_prf(results, logger)
        compute_sentence_level_prf(results, logger)

        logger.info('>>>> SIGHAN15')
        results = []
        for batch in self.sighan15:
            out = self.forward(batch)
            pre_c, pre_d = out[3].tolist(), out[2].tolist()
            src, trg, d_labels = batch['input_ids'].tolist(), batch['labels'].tolist(), batch['pos_labels'].tolist()
            l = batch['len'].tolist()
            for p_c, p_d, s, t, d_l, ll in zip(pre_c, pre_d, src, trg, d_labels, l):
                p_d = p_d[1:ll+1]
                s = s[1:ll+1]
                t = t[1:ll+1]
                p_c = p_c[1:ll+1]
                d_l = d_l[1:ll+1]
                results.append((s, p_c, t, p_d, d_l))
        # self.compute_corrector_prf(results)
        compute_corrector_prf(results, logger)
        compute_sentence_level_prf(results, logger)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        logger.info('Test.')
        self.validation_epoch_end(outputs)
    
    
    def compute_corrector_prf(self, results):
        # src, pre, trg, pro_d, trg_d
        ll = [0.5, 0.6, 0.7, 0.8, 0.9]
        acc, pre, rec, f1 = 0, 0, 0, 0
        acc_s, pre_s, rec_s, f1_s = 0, 0, 0, 0
        for p in ll:
            out = self.compute_detect(results, p)
            if out[-1] > f1_s:
                acc, pre, rec, f1, acc_s, pre_s, rec_s, f1_s = out
        logger.info(
        "The character detection result is acc={}, precision={}, recall={} and F1={}".format(acc, pre, rec, f1))
        logger.info(
        "The sentence detection result is acc={}, precision={}, recall={} and F1={}".format(acc_s, pre_s, rec_s, f1_s))

        tp, fp, tn, fn = 0, 0, 0, 0
        tp_s, fp_s, tn_s, fn_s = 0, 0, 0, 0

        for item in results:
            src, pre, trg, _, _ = item
            for s, p, t in zip(src, pre, trg):
                if s != t:
                    if p == t:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if p == t:
                        tn += 1
                    else:
                        fp += 1
            if src != trg:
                if pre == trg:
                    tp_s += 1
                else:
                    fn_s += 1
            else:
                if pre == trg:
                    tn_s += 1
                else:
                    fp_s += 1
        TP, TN, FP, FN = tp, tn, fp, fn
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (pre * rec) / (pre + rec)
    
        logger.info(
        "The character correction result is acc={}, precision={}, recall={} and F1={}".format(acc, pre, rec, f1))
        
        TP, TN, FP, FN = tp_s, tn_s, fp_s, fn_s
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (pre * rec) / (pre + rec)
        logger.info(
        "The sentence correction result is acc={}, precision={}, recall={} and F1={}".format(acc_s, pre_s, rec_s, f1_s))

    
    def compute_corrector_prf0(self, results):
        # src, pre, trg, pro_d, trg_d
        ll = [0.5, 0.6]
        acc, pre, rec, f1 = 0, 0, 0, 0
        acc_s, pre_s, rec_s, f1_s = 0, 0, 0, 0
        for p in ll:
            out = self.compute_detect(results, p)
            if out[-1] > f1_s:
                acc, pre, rec, f1, acc_s, pre_s, rec_s, f1_s = out
        logger.info(
        "The character detection result is acc={}, precision={}, recall={} and F1={}".format(acc, pre, rec, f1))
        logger.info(
        "The sentence detection result is acc={}, precision={}, recall={} and F1={}".format(acc_s, pre_s, rec_s, f1_s))

        tp, fp, tn, fn = 0, 0, 0, 0
        tp_s, fp_s, tn_s, fn_s = 0, 0, 0, 0

        for item in results:
            src, pre, trg, _, _ = item
            tp0, fp0, tn0, fn0 = 0, 0, 0, 0
            for s, p, t in zip(src, pre, trg):
                if s == t:
                    if p == s:
                        tn0 += 1
                    else:
                        fp0 += 1
                else:
                    if p == t:
                        tp0 += 1
                    else:
                        fn0 += 1
                        
            if fn0 + tp0 > 0:
                if fn0 + fp0 == 0:
                    tp_s += 1
                else:
                    fn_s += 1
            else:
                if fn0 + fp0 == 0:
                    tn_s += 1
                else:
                    fp_s += 1
            tp, fp, tn, fn = tp + tp0, fp + fp0, tn + tn0, fn + fn0

        TP, TN, FP, FN = tp, tn, fp, fn
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (pre * rec) / (pre + rec)
    
        logger.info(
        "The character correction result is acc={}, precision={}, recall={} and F1={}".format(acc, pre, rec, f1))
        
        TP, TN, FP, FN = tp_s, tn_s, fp_s, fn_s
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (pre * rec) / (pre + rec)
        logger.info(
        "The sentence correction result is acc={}, precision={}, recall={} and F1={}".format(acc_s, pre_s, rec_s, f1_s))
    def compute_detect(self, results, pro):
        tp, fp, tn, fn = 0, 0, 0, 0
        tp_s, fp_s, tn_s, fn_s = 0, 0, 0, 0

        for item in results:
            _, _, _, pre, trg = item
            ss = ''
            for p, t in zip(pre, trg):
                if t == 0:
                    if p < pro:
                        tn += 1
                        ss += '0'
                    else:
                        fp += 1
                        ss += '1'
                        
                else:
                    if p >= pro:
                        tp += 1
                        ss += '1'
                    else:
                        fn += 1
                        ss += '0'
                        
            trg = ''.join([str(it) for it in trg])
            if '1' in list(trg):
                if ss == trg:
                    tp_s += 1
                else:
                    fn_s += 1
            else:
                if pre == trg:
                    tn_s += 1
                else:
                    fp_s += 1
        TP, TN, FP, FN = tp, tn, fp, fn
        acc_c = (TP+TN)/(TP+TN+FP+FN)
        pre_c = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec_c = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre_c + rec_c == 0:
            f1_c = 0
        else:
            f1_c = 2 * (pre_c * rec_c) / (pre_c + rec_c)
        
        TP, TN, FP, FN = tp_s, tn_s, fp_s, fn_s
        acc_s = (TP+TN)/(TP+TN+FP+FN)
        pre_s = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec_s = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre_s + rec_s == 0:
            f1_s = 0
        else:
            f1_s = 2 * (pre_s * rec_s) / (pre_s + rec_s)
        
        return acc_c, pre_c, rec_c, f1_c, acc_s, pre_s, rec_s, f1_s
    
    def compute_detect0(self, results, pro):
        tp, fp, tn, fn = 0, 0, 0, 0
        tp_s, fp_s, tn_s, fn_s = 0, 0, 0, 0

        for item in results:
            _, _, _, pre, trg = item
            tp0, fp0, tn0, fn0 = 0, 0, 0, 0
            for p, t in zip(pre, trg):
                if t == 0:
                    if p < pro:
                        tn0 += 1
                    else:
                        fp0 += 1
                else:
                    if p >= pro:
                        tp0 += 1
                    else:
                        fn0 += 1
                        
            if fn0 + tp0 > 0:
                if fn0 + fp0 == 0:
                    tp_s += 1
                else:
                    fn_s += 1
            else:
                if fn0 + fp0 == 0:
                    tn_s += 1
                else:
                    fp_s += 1
            tp, fp, tn, fn = tp + tp0, fp + fp0, tn + tn0, fn + fn0

        TP, TN, FP, FN = tp, tn, fp, fn
        acc_c = (TP+TN)/(TP+TN+FP+FN)
        pre_c = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec_c = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre_c + rec_c == 0:
            f1_c = 0
        else:
            f1_c = 2 * (pre_c * rec_c) / (pre_c + rec_c)
        
        TP, TN, FP, FN = tp_s, tn_s, fp_s, fn_s
        acc_s = (TP+TN)/(TP+TN+FP+FN)
        pre_s = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec_s = TP / (TP + FN) if (TP + FN) > 0 else 0
        if pre_s + rec_s == 0:
            f1_s = 0
        else:
            f1_s = 2 * (pre_s * rec_s) / (pre_s + rec_s)
        
        return acc_c, pre_c, rec_c, f1_c, acc_s, pre_s, rec_s, f1_s
