import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        #get the model from bert path
        self.bert=transformers.BertModel.from_pretrained(config.BERT_PATH)
        #add a dropout for regularization
        self.bert_drop=nn.Dropout(0.3)
        #a simple logictic classification problem
        self.out=nn.Linear(768,1)

    def forward(self,ids,mask,token_type_ids):
        ## BERT in its default setting returns two outputs last
        ## hidden state and output of the bert pooler layer
        ## we use the output of the pooler layer which is of the size (batch_size,hidden_size)
        ## hidden size =768/1024 depending on bert base of bert large
        ## we can use several hidden layers also

        _,o2=self.bert(ids,
                       attention_mask=mask,
                       token_type_ids=token_type_ids
                       )
        #pass through the dropout layer
        bo=self.bert_drop(o2)
        output=self.out(bo)
        return output
    
