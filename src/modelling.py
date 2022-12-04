from torch import nn
import torch
from transformers import BertTokenizer, AutoModel, BertForMaskedLM
from transformers import ElectraForPreTraining
import math
import torch.nn.functional as F
from transformer import TransformerBlock  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class DemoModel(nn.Module):
    def __init__(self, path):
        super(DemoModel, self).__init__()
        # self.bert = AutoModel.from_pretrained(path)
        self.bert = AutoModel.from_pretrained(path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 21128)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=0)
        self.sigmoid = nn.Sigmoid()
        self.nllloss_func=nn.NLLLoss()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # output = self.bert(input_ids, token_type_ids, attention_mask).logits
        output = self.bert(input_ids, token_type_ids, attention_mask)
        output = self.classifier(output.last_hidden_state)
        # output = torch.log(self.sigmoid(output))
        if labels is not None:
            loss = self.CrossEntropyLoss(output.view(-1, self.bert.config.vocab_size), labels.view(-1))
            # loss = self.nllloss_func(output.view(-1, self.bert.config.vocab_size), labels.view(-1))
            return loss
        else:
            return output

class DetectModel(nn.Module):
    def __init__(self, DetectModelPath, type='electra'):
        super(DetectModel, self).__init__()
        if type == 'bert':
            self.bert = AutoModel.from_pretrained(DetectModelPath)
        else:
            self.bert = ElectraForPreTraining.from_pretrained(DetectModelPath)
        self.type = type
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, **kwarg):
        output = self.bert(input_ids, token_type_ids, attention_mask, labels)
        loss = None
        if self.type == 'bert':
            output = self.classifier(output.last_hidden_state)
            s_output = output.size()
            output = output.view(s_output[0], -1)
        else:
            output = output.logits
            loss = output.loss
        # output:[batch, seq_len]
        return loss, output

class SemanticModel(nn.Module):
    def __init__(self, SemanticModelPath, type='electra'):
        super(SemanticModel, self).__init__()
        if type == 'bert':
            self.bert = AutoModel.from_pretrained(SemanticModel)
        else:
            self.bert = ElectraForPreTraining.from_pretrained(SemanticModel)
        self.type = type

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, **kwarg):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        if self.type == 'bert':
            output = output.last_hidden_state
        else:
            output = output.logits
        # output:[batch, seq_len, hid]
        return output

# TODO(@马承成): 完成时间2022/09/10
class SpeechModel(nn.Module):
    def __init__(self, SpeechModelPath, type='bert'):
        super(SpeechModel, self).__init__()
        pass
    
    
# TODO(@马承成): 完成时间2022/09/10
class GlyphModel(nn.Module):
    def __init__(self, GlyphModelPath, type='bert'):
        super(GlyphModel, self).__init__()
        pass
    

class CorrectorEncoder(nn.Module):
    def __init__(
            self,
            SemanticModelPath,
            SpeechModelPath,
            GlyphModelPath,
            typeSemantic='bert',
            typeSpeech='',
            typeGlyph='bert'
            ):
        
        self.GlyphModel = GlyphModel(GlyphModelPath, typeGlyph)
        self.SpeechModel = SpeechModel(SpeechModelPath, typeSpeech)
        self.SemanticModel = SemanticModel(SemanticModelPath, typeSemantic)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        
        self.transformer1 = TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        self.transformer2 =  TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        self.transformer3 = TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        
        self.sigmoid = nn.Sigmoid()
        
        self.cls = nn.Linear(self.bert.config.hidden_size, 21128)
        
    def getKL4AttentionMatrix(self, l1, l2, l3, mask, if_print_attention_matrix):
        batch_size = l1.size(0)
        hid = l1.size(-1)
        
        if mask is not None:
            mask = torch.matmul(mask, mask.transpose(-2, -1))
            
        scores12 = torch.matmul(l1, l2.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores13 = torch.matmul(l1, l3.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores23 = torch.matmul(l2, l3.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores21 = torch.matmul(l2, l1.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores31 = torch.matmul(l3, l1.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores32 = torch.matmul(l3, l2.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        
        if if_print_attention_matrix:
            torch.save(scores12.to(torch.device('cpu')), 'scores12')
            torch.save(scores13.to(torch.device('cpu')), 'scores13')
            torch.save(scores23.to(torch.device('cpu')), 'scores23')
            torch.save(scores21.to(torch.device('cpu')), 'scores21')
            torch.save(scores31.to(torch.device('cpu')), 'scores31')
            torch.save(scores32.to(torch.device('cpu')), 'scores32')

        if mask is not None:
            scores12 = scores12.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores13 = scores13.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores23 = scores23.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores21 = scores21.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores31 = scores31.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores32 = scores32.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)

        input = self.logSoftMax(torch.cat((scores12, scores13, scores23, scores21, scores31, scores32), dim=0))
        kloss = 0
        for l in [scores12, scores13, scores23, scores21, scores31, scores32]:
            target = self.softMax(l)
            # target = torch.cat((l, l, l, l, l, l), dim=0)
            kloss += self.kl_loss(input, target)
        
        return kloss
    
    
    def getInfoNCELoss(self, l1, l2, l3, mask, if_print_attention_matrix):
        shape = l1.shape
        if mask is not None:
            mask = mask.view(-1, shape[-1])
            mask = torch.cat((mask, mask, mask), dim=0)
            mask = torch.matmul(mask, mask.transpose(-2, -1))
        
        l1 = l1.view(-1, shape[-1])
        l2 = l2.view(-1, shape[-1])
        l3 = l3.view(-1, shape[-1])
        
        l123 = torch.cat((l1, l2, l3), dim=0)
        if mask is not None:
            l123 = l123.masked_fill(mask == 0, -1e9)
        
        
        y1 = torch.arange(l123.shape[0])
        y2 = torch.arange(l123.shape[0])
        y1 = (y1 + l1.shape[0]) %  (3 * l1.shape[0])
        y2 = (y2 + 2 * l1.shape[0]) %  (3 * l1.shape[0])
        
        
        loss = 0
        sim = F.cosine_similarity(l123.unsqueeze(1), l123.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(sim.shape[0]) * 1e12
        sim = sim / 0.05
        
        loss1 = F.cross_entropy(sim, y1)
        loss2 = F.cross_entropy(sim, y2)
        loss = loss1 + loss2
        
        return loss
    
    def forward(self, x, if_print_attention_matrix=False):
        logits4Semantic = self.SemanticModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_token_type_ids'], attention_mask=x['semantic_attention_mask'])
        logits4Glyph = self.GlyphModel(input_ids=x['glyph_input_ids'], token_type_ids=x['glyph_token_type_ids'], attention_mask=x['glyph_attention_mask'])
        # TODO(@马承成)
        logits4Speech = self.SpeechModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_input_ids'], attention_mask=x['semantic_input_ids'])
        
        KLoss = self.getKL4AttentionMatrix(logits4Semantic, logits4Glyph, logits4Speech, x['semantic_attention_mask'], if_print_attention_matrix)
        infoNCELoss = self.getInfoNCELoss(logits4Semantic, logits4Glyph, logits4Speech, x['semantic_attention_mask'], if_print_attention_matrix)
        
        
        return logits4Semantic, logits4Glyph, logits4Speech, KLoss+infoNCELoss
    
class Corrector(nn.Module):
    def __init__(
            self,
            SemanticModelPath,
            SpeechModelPath,
            GlyphModelPath,
            DetectModelPath,
            typeSemantic='bert',
            typeSpeech='',
            typeGlyph='bert',
            typeDetect='electra',
            if_KLoss=True,
            if_InfoNCELoss=True,
            if_pretrain=False
            ):
        
        self.GlyphModel = GlyphModel(GlyphModelPath, typeGlyph)
        self.DetectModel = DetectModel(DetectModelPath, typeDetect)
        self.SpeechModel = SpeechModel(SpeechModelPath, typeSpeech)
        self.SemanticModel = SemanticModel(SemanticModelPath, typeSemantic)
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=0)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        
        self.if_pretrain = if_pretrain
        self.if_KLoss = if_KLoss
        self.if_InfoNCELoss = if_InfoNCELoss
        
        self.transformer1 = TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        self.transformer2 = TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        self.transformer3 = TransformerBlock(hidden=self.bert.config.hidden_size, attn_heads=8, dropout=0.3)
        
        self.sigmoid = nn.Sigmoid()
        self.cls = nn.Linear(self.bert.config.hidden_size, 21128)
        self.cls4GlyphModel = nn.Linear(self.bert.config.hidden_size, 21128)
        self.cls4SpeechModel = nn.Linear(self.bert.config.hidden_size, 21128)
        self.cls4SemanticModel = nn.Linear(self.bert.config.hidden_size, 21128)
    
    def getKL4AttentionMatrix(self, l1, l2, l3, mask, if_print_attention_matrix):
        batch_size = l1.size(0)
        hid = l1.size(-1)
        
        if mask is not None:
            mask = torch.matmul(mask, mask.transpose(-2, -1))
            
        scores12 = torch.matmul(l1, l2.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores13 = torch.matmul(l1, l3.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores23 = torch.matmul(l2, l3.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores21 = torch.matmul(l2, l1.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores31 = torch.matmul(l3, l1.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        scores32 = torch.matmul(l3, l2.transpose(-2, -1)) / math.sqrt(l1.size(-1))
        
        if if_print_attention_matrix:
            torch.save(scores12.to(torch.device('cpu')), 'scores12')
            torch.save(scores13.to(torch.device('cpu')), 'scores13')
            torch.save(scores23.to(torch.device('cpu')), 'scores23')
            torch.save(scores21.to(torch.device('cpu')), 'scores21')
            torch.save(scores31.to(torch.device('cpu')), 'scores31')
            torch.save(scores32.to(torch.device('cpu')), 'scores32')

        if mask is not None:
            scores12 = scores12.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores13 = scores13.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores23 = scores23.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores21 = scores21.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores31 = scores31.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)
            scores32 = scores32.masked_fill(mask == 0, -1e9).view(batch_size, hid * hid)

        input = self.logSoftMax(torch.cat((scores12, scores13, scores23, scores21, scores31, scores32), dim=0))
        kloss = 0
        for l in [scores12, scores13, scores23, scores21, scores31, scores32]:
            target = self.softMax(l)
            # target = torch.cat((l, l, l, l, l, l), dim=0)
            kloss += self.kl_loss(input, target)
        
        return kloss
    
    
    def getInfoNCELoss(self, l1, l2, l3, mask, if_print_attention_matrix):
        shape = l1.shape
        if mask is not None:
            mask = mask.view(-1, shape[-1])
            mask = torch.cat((mask, mask, mask), dim=0)
            mask = torch.matmul(mask, mask.transpose(-2, -1))
        
        l1 = l1.view(-1, shape[-1])
        l2 = l2.view(-1, shape[-1])
        l3 = l3.view(-1, shape[-1])
        
        l123 = torch.cat((l1, l2, l3), dim=0)
        if mask is not None:
            l123 = l123.masked_fill(mask == 0, -1e9)
        
        
        y1 = torch.arange(l123.shape[0])
        y2 = torch.arange(l123.shape[0])
        y1 = (y1 + l1.shape[0]) %  (3 * l1.shape[0])
        y2 = (y2 + 2 * l1.shape[0]) %  (3 * l1.shape[0])
        
        
        loss = 0
        sim = F.cosine_similarity(l123.unsqueeze(1), l123.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(sim.shape[0]) * 1e12
        sim = sim / 0.05
        
        loss1 = F.cross_entropy(sim, y1)
        loss2 = F.cross_entropy(sim, y2)
        loss = loss1 + loss2
        
        return loss
        
        
    def getDetectLoss(self, x, label):
        pass

    def forward(self, x, if_print_attention_matrix=False):
        logits4Semantic = self.SemanticModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_token_type_ids'], attention_mask=x['semantic_attention_mask'])
        logits4Glyph = self.GlyphModel(input_ids=x['glyph_input_ids'], token_type_ids=x['glyph_token_type_ids'], attention_mask=x['glyph_attention_mask'])
        # TODO(@马承成)
        logits4Speech = self.SpeechModel(input_ids=x['semantic_input_ids'], token_type_ids=x['semantic_input_ids'], attention_mask=x['semantic_input_ids'])
        detect_loss, logits4Detect = self.DetectModel(input_ids=x['detector_input_ids'], token_type_ids=x['detector_token_type_ids'], attention_mask=x['detector_attention_mask'], labels=x['detect_label'])
        
        KLoss = self.getKL4AttentionMatrix(logits4Semantic, logits4Glyph, logits4Speech, x['semantic_attention_mask'], if_print_attention_matrix)
        infoNCELoss = self.getInfoNCELoss(logits4Semantic, logits4Glyph, logits4Speech, x['semantic_attention_mask'], if_print_attention_matrix)
        
        if self.if_pretrain:
            logits4Semantic = self.cls4SemanticModel(logits4Semantic)
            logits4Glyph = self.cls4GlyphModel(logits4Glyph)
            logits4Speech = self.cls4SpeechModel(logits4Speech)
            
            loss = self.CrossEntropyLoss(logits4Semantic.transpose(1, 2), x['tar_label'])
            loss += self.CrossEntropyLoss(logits4Glyph.transpose(1, 2), x['tar_label'])
            loss += self.CrossEntropyLoss(logits4Speech.transpose(1, 2), x['tar_label'])
            
            return logits4Semantic, logits4Glyph, logits4Speech, 0.3*KLoss+0.3*infoNCELoss+0.4*loss
        # logits_in: batch * seqlen * hid
        else:
            shape = logits4Semantic.shape
            logits_in = torch.cat((logits4Semantic, logits4Glyph, logits4Speech), dim=1)
            
            logits_m, att1 = self.transformer1(logits_in, x['semantic_input_ids'])
            logits_m, att2 = self.transformer2(logits_m, x['semantic_input_ids'])
            logits_out, att3 = self.transformer3(logits_m, x['semantic_input_ids'])
            
            if if_print_attention_matrix:
                torch.save(att1.to(torch.device('cpu')), 'att1')
                torch.save(att2.to(torch.device('cpu')), 'att2')
                torch.save(att3.to(torch.device('cpu')), 'att3')
            
            P = self.sigmoid(logits4Detect)
            logits = logits_out * P + logits_in * (1 - P)
            logits4Semantic = logits[:, :shape[1], :] + logits4Semantic
            logits4Glyph = logits[:, shape[1]:shape[1]*2, :] + logits4Glyph
            logits4Speech = logits[:, shape[1]*2:, :] + logits4Speech
            logits = logits4Semantic + logits4Glyph + logits4Speech
            
            logits = self.cls(logits)
            
            loss = detect_loss + infoNCELoss + KLoss
            
            loss += self.CrossEntropyLoss(logits.transpose(1, 2), x['tar_label'])
            return logits, loss
    
def test_detect():
    from transformers import ElectraForPreTraining, ElectraTokenizerFast
    import torch

    # discriminator = DetectModel("hfl/chinese-electra-180g-large-discriminator")
    discriminator = DetectModel('bert-base-chinese', 'bert')
    # discriminator = ElectraForPreTraining.from_pretrained("hfl/chinese-electra-180g-large-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("hfl/chinese-electra-180g-large-discriminator")

    sentence = "The quick brown fox jumps over the lazy dog"
    fake_sentence = [["我爱背景天安门","我爱背景天安门我爱背景天安门"],["我爱背景天安门","我爱背景天安门我爱背景天安门"]]

    fake_tokens = tokenizer(fake_sentence,return_tensors="pt")
    # fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    
    # print(fake_tokens)
    
    discriminator_outputs = discriminator(fake_tokens.input_ids, fake_tokens.token_type_ids, fake_tokens.attention_mask)
    print(discriminator_outputs)
    predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

    print(predictions)
    # [print("%7s" % token, end="") for token in fake_tokens]
    # print()
    # [print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]
    # print()

if __name__ == "__main__":
    test_detect()