from torch.utils.data import Dataset
import tqdm
import torch
import random

# 23个声母，24个韵母
initials = 'b p m f d t n l ɡ k h j q x zh ch sh r z c s y w'.split(' ')
initial2id = {initial:i+1 for i, initial in enumerate(initials)}

finals = 'a o e i u v ai ei ui ao ou iu ie ue er an en in un vn ang eng ing ong'.split(' ')
final2id = {final:i for i, final in enumerate(finals)}

from pypinyin import lazy_pinyin, Style
# print(lazy_pinyin('绿色，',style = Style.TONE3))




def getPictures(char, savePath, fontType, fontPath):
    from PIL import Image, ImageFont, ImageDraw
    image = Image.new('RGB', (250, 250), (255,255,255)) 
    iwidth, iheight = image.size
    font = ImageFont.truetype(fontPath+'/'+fontType, 60)
    draw = ImageDraw.Draw(image)

    fwidth, fheight = draw.textsize(char, font)

    fontx = (iwidth - fwidth - font.getoffset(char)[0]) / 2
    fonty = (iheight - fheight - font.getoffset(char)[1]) / 2

    draw.text((fontx, fonty), char, 'black', font)
    image.save(savePath+char+'.jpg') 

class MultiModalDataset(Dataset):
    def __init__(self, path, if_training=False, maxlen=128, glyphTokenizer=None, semanticTokenizer=None, speechTokenizer=None, detectorTokenizer=None, if_pretrain=False):
        self.maxlen = maxlen
        self.glyphTokenizer = glyphTokenizer
        self.semanticTokenizer = semanticTokenizer
        self.speechTokenizer = speechTokenizer
        self.detectorTokenizer = detectorTokenizer
        self.if_pretrain = if_pretrain
        # pre_train: src, src
        # finetune: src, tar, pos_label
        
        self.corpus = []
        with open(path, 'r') as f:
            for line in f.readlines():
                # src tar detect_label
                line = line.strip().split('\t')
                self.corpus.append(line)
        self.if_training = if_training
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        if not self.if_pretrain:
            text_src, text_tar, detect_label = self.corpus[idx]
            len_src = len(text_src)
        else:
            text_src = self.corpus[idx]
            text_tar = text_src[:]
        
        ret = {}
        
        # Glyph Tokenizer
        glyphInput = self.getGlyphData(text_src)
        ret['glyph_input_ids'] = glyphInput['input_ids']
        ret['glyph_token_type_ids'] = glyphInput['token_type_ids']
        ret['glyph_attention_mask'] = glyphInput['attention_mask']
        
        # Semantic Tokenizer
        semanticInput = self.getSemanticData(text_src)
        ret['semantic_input_ids'] = semanticInput['input_ids']
        ret['semantic_token_type_ids'] = semanticInput['token_type_ids']
        ret['semantic_attention_mask'] = semanticInput['attention_mask']
        
        # Speech Tokenizer
        # TODO(@马承成): 完成时间(2022/09/10)
        
        # Detector Tokenizer
        detectorInput = self.getDetectorData(text_src)
        ret['detector_input_ids'] = detectorInput['input_ids']
        ret['detector_token_type_ids'] = detectorInput['token_type_ids']
        ret['detector_attention_mask'] = detectorInput['attention_mask']
        
        # label
        # text_tar and detect_label
        label = self.getLabelTar(text_tar)
        ret['tar_label'] = label['tar_label']
        if not self.if_pretrain:
            ret['detect_label'] = self.getLabelPos(detect_label)['detect_label']
        return ret
    
    def getLabelTar(self, text_tar):
        tar = self.semanticTokenizer(
            text_tar,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )
        ret = {}
        ret['tar_label'] = tar['input_ids']
        return ret
    
    def getLabelPos(self, detect_label):
        ret = {}
        detect_label = [int(label) for label in detect_label]
        ret['detect_label'] = torch.tensor(detect_label, dtype=torch.long)
        return ret

    def getSpeechData(self, text_src):
        # TODO(@马承成)：完成时间(2022/09/10)
        pass
    
    def getGlyphData(self, text_src):
        return self.glyphTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )
    
    def getSemanticData(self, text_src):
        return self.semanticTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )

    def getDetectorData(self, text_src):
        return self.detectorTokenizer(
            text_src,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )
        
    
class dataset_demo:
    def __init__(self, path, if_training=False, maxlen=128, Tokenizer=None):
        self.maxlen = maxlen
        self.tokenizer = Tokenizer
        # pre_train: src, src
        # finetune: src, tar, pos_label
        
        self.corpus = []
        with open(path, 'r') as f:
            for line in f.readlines():
                # src tar detect_label
                line = line.strip().split('\t')
                self.corpus.append(line)
        self.if_training = if_training
        self.len = len(self.corpus)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        item = self.corpus[index]
        src, tar, pos_label = item[0], item[1], item[2]
        ret = {}
        demo_input = self.tokenizer(
            src,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )
        demo_tar_label = self.tokenizer(
            tar,
            add_special_tokens=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen
        )
        ret['input_ids'] = demo_input['input_ids'].view(-1)
        ret['token_type_ids'] = demo_input['token_type_ids'].view(-1)
        ret['attention_mask'] = demo_input['attention_mask'].view(-1)
        ret['tar_label'] = demo_tar_label['input_ids'].view(-1)
        ret['len'] = torch.tensor(len(tar), dtype=torch.long)
        detect_label = []
        for i, j in zip(ret['input_ids'], ret['tar_label']):
            if i == j:
                detect_label.append(0)
            else:
                detect_label.append(1)
        ret['detect_label'] = torch.tensor(detect_label, dtype=torch.long)
        return ret

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained('hfl/chinese-macbert-large')
    train_path = '/var/zgcCorrector/data/src/train13.txt'
    train_data = dataset_demo(train_path, Tokenizer=tk)
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                         batch_size=64,
    #                                         num_workers=8,
    #                                         shuffle=True)
    for i in range(train_data.__len__()):
        if train_data.__getitem__(i)['tar_label'].shape[0] != 128:
            print(i)
            break