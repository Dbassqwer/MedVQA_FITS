import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from CoT.cotnet import se_cotnetd_152, se_cotnetd_152_L
from transformers import BertModel
from transformers import RobertaModel
from transformers import ElectraModel


class FITS_Model(nn.Module):
    def __init__(self, args):
        super(FITS_Model, self).__init__()
        self.args = args
        self.transformer = Transformer(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))

    def forward(self, img, input_ids, segment_ids, input_mask, point_token, point_segment_ids):
        h, cache_h = self.transformer(img, input_ids, segment_ids, input_mask, point_token, point_segment_ids)

        h0 = self.fc1(h.mean(0).mean(1))

        h1 = self.activ1(h0)

        logits = self.classifier(h1)

        return logits


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.TI = TI_module(args)
        self.TQ = TQ_module(args)
        self.blocks = BertLayer(args, norm='pre')
        self.n_layers = args.n_layers

    def forward(self, img, input_ids, token_type_ids, mask, point_token, point_segment_ids):
        v_e = self.TI(img)
        tq_e = self.TQ(input_ids, token_type_ids, point_token, point_segment_ids, v_e)

        hidden_states = []
        cache_tq_e = tq_e
        for i in range(self.n_layers):
            hid, attn_scores = self.blocks(tq_e, mask, i)
            hidden_states.append(hid)

        return torch.stack(hidden_states, 0), cache_tq_e


class TI_module(nn.Module):
    def __init__(self, args):
        super(TI_module, self).__init__()

        self.args = args
        self.hidden_size = args.hidden_size

        if args.img_backbone == 'resnet':
            self.model = models.resnet152(pretrained=True)
        elif args.img_backbone == 'cotnet':  # size 224*224
            self.model = se_cotnetd_152(num_classes=3)
            self.model.load_state_dict(torch.load(args.cot_dir + '/bio_se_cotnetd_152.pth.tar'))
        elif args.img_backbone == 'cotnets':
            self.model = se_cotnetd_152_L()  # size 320*320
            self.model.load_state_dict(torch.load(args.cot_dir + '/bio_se_cotnetd_L_152.pth.tar'))

        # the image encoder parameters can be trained
        # for p in self.parameters():
        #     p.requires_grad = False

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(2048, self.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img):
        modules_slice = list(self.model.children())[:-2]
        img_backbone = nn.Sequential(*modules_slice)
        img1 = img_backbone(img)
        img2 = self.conv(img1)
        img3 = self.relu(img2)
        img_v = self.pool(img3)
        img_v = img_v.view(-1, self.hidden_size)
        return img_v  # torch.Size([b, hidden_size])


class TQ_module(nn.Module):
    def __init__(self, args):
        super(TQ_module, self).__init__()
        self.args = args
        bert_type = args.bert_type
        if bert_type == 'bert':
            base_model = BertModel.from_pretrained('bert-base-uncased')
        elif bert_type == 'blue_bert':
            base_model = BertModel.from_pretrained(args.bbert_dir)
        elif bert_type == 'roberta':
            base_model = RobertaModel.from_pretrained('roberta-base')
        elif bert_type == 'biom':
            base_model = ElectraModel.from_pretrained(args.bbert_dir)
        bert_model = nn.Sequential(*list(base_model.children())[0:])

        self.bert_embedding = bert_model[0]

        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        if args.move_v_type == 'all':
            self.LayerNorm2 = nn.LayerNorm(args.hidden_size, eps=1e-5)
            self.dropout2 = nn.Dropout(args.hidden_dropout_prob)

            self.LayerNorm3 = nn.LayerNorm(args.hidden_size, eps=1e-5)
            self.dropout3 = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, point_token, point_segment_ids, v_e):

        q_e = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        tp_e = self.bert_embedding(input_ids=point_token, token_type_ids=point_segment_ids, position_ids=None)

        if self.args.move_v_type == 'no' or self.args.move_v_type == 'all':
            tp_e_trans = tp_e.mean(1).unsqueeze(1).expand(-1, self.args.max_position_embeddings, self.args.hidden_size)
            tq_e = q_e + tp_e_trans
            tq_e = self.LayerNorm(tq_e)
            tq_e = self.dropout(tq_e)
            q_e = tq_e
        for i in range(len(q_e)):  # replace image feature embeddings
            if self.args.move_v_type == 'all':
                q_e[i][1] = self.dropout3(self.LayerNorm3(v_e[i] + tp_e.mean(1)[i]))
            else:
                q_e[i][1] = v_e[i]

        if self.args.move_v_type == 'less':
            tp_e_trans = tp_e.mean(1).unsqueeze(1).expand(-1, self.args.max_position_embeddings, self.args.hidden_size)
            tq_e = q_e + tp_e_trans
            tq_e = self.LayerNorm(tq_e)
            tq_e = self.dropout(tq_e)
            q_e = tq_e

        return q_e


class BertLayer(nn.Module):
    def __init__(self, args, norm='pre'):
        super(BertLayer, self).__init__()
        self.norm_pos = norm  # pre-LN or post-LN
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-5)
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
        self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
        self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])

    def forward(self, hidden_states, attention_mask, layer_num):
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                attn_output, attn_scores = self.attention[layer_num](self.norm1(hidden_states), attention_mask)
                h = self.proj[layer_num](attn_output)
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadedSelfAttention, self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads

    def forward(self, x, mask):  # hidden_states, attention_mask
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h, scores

    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, args):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.fc2 = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

    def gelu(self, x):
        import math
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
