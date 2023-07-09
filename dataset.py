#dataset.py

import config
import torch

class BERTDataset:
    def __init__(self,review,target):
        self.review=review
        self.target=target

        #we fetch the tokenizer and max len from config file
        self.tokenizer=config.TOKENIZER
        self.max_len=config.MAX_LEN

    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,item):
        #for a given item return a dictionary of inputs
        review=str(self.review[item])
        review=" ".join(review.split())
        #encode plus comes from huggingface transformers and exists for all tokenizerss they offer it can be used
        # to convert a given string to ids mask and token type ids which are needed for models like bert
        #her review is a stirng

        inputs=self.tokenizer.encode_plus(
            review,
            None,
            add_special_token=True,
            max_length=self.max_len,
            pad_to_max_len=True,
        )
        #ids are ids of tokens generated after tokenizing reviews
        ids,mask=inputs["input_ids"],inputs["attention_mask"]
        token_type_ids=inputs["token_type_ids"]

        return{
            "ids":torch.tensor(ids,dtype=torch.long),
            "mask":torch.tensor(mask,dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
            "targets":torch.tensor(self.target[item],dtype=torch.float)
        }

