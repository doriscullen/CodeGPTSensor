import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)
    
    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),output_hidden_states=True)

        token_embeddings = out[0]

        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1) 
        sentence_embeddings = sentence_embeddings

        return sentence_embeddings

    def forward(self, input_ids, contrast_ids=None, labels=None):
        
        vec = self.get_xcode_vec(input_ids)
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits, dim=-1)

        # Cross Entropy Loss
        loss = F.cross_entropy(logits, labels)
        
        if self.args.contrast and self.args.do_train:
            # Dropout twice to get its positive equivalent
            vec2 = self.get_xcode_vec(input_ids)
            logits2 = self.classifier(vec2)
            kl_loss = get_kl_loss(logits, logits2) 

            # The negative sample
            vec3 = self.get_xcode_vec(contrast_ids)
            contrast_loss = get_contrast_loss(vec, vec3, labels,device=self.args.device) 
            
            # Final loss
            loss = loss + 0.1 * kl_loss + 0.2 * contrast_loss
   
        return loss, prob


def get_contrast_loss(vec, contrast, label, device):
    label_new = torch.full(label.size(), -1, device=device)
    loss = F.cosine_embedding_loss(vec, contrast, label_new)
    return loss


def get_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss