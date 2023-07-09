import config
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dataset
import engine

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def train():
    dfx=pd.read_csv(config.TRAINING_FILE).fillna("none")
    dfx.sentiment.apply(lambda x: 1 if "positive" else 0)
    df_train,df_valid=model_selection.train_test_split(dfx,test_size=0.1,seed=42,stratify=dfx.sentiment.values)

    ## reset the index
    df_train=df_train.reset_index(drop=True)
    df_test=df_test.reset_index(drop=True)

    #initialize the bert dataset
    train_dataset=dataset.BERTDataset(reveiew=df_train.review.values,
                                      target=df_train.sentiment.values
                                      )
    train_data_loader=torch.utils.data.DataLoader(train_dataset,config.TRAIN_BATCH_SIZE,num_workers=4)
    
    
    valid_dataset=dataset.BERTDataset(review=df_valid.review.values,target=df_valid.sentiment.values)
    valid_data_loader=torch.utils.data.DataLoader(valid_dataset,config.VALID_BATCH_SIZE,num_workers=4)

    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model=BERTBaseUncased()
    model.to(device)

    param_optimizer=list(model.named_parameters())
    no_decay=["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters=[{
        "params":[
            p for n,p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay":0.001,
        },
        {
        "params":[
            p for n,p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay":0.0,
        },
    ]
    # calculate the number of training steps this is used by scheduler
    num_train_steps=int(len(df_train)/config.TRAIN_BATCH_SIZE *config.EPOCHS)

    optimizer=AdamW(optimizer_parameters,lr=3e-5)

    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    ## if you have multiple GPUs model to dataParallel to use multiple GPUs
    model=nn.DataParallel(model)

    best_accuracy=0.0
    ## start the training epochs
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader,model,optimizer,device,scheduler)
        outputs,targets=engine.eval_fn(valid_data_loader,model,device)

        outputs=np.array(outputs)>=0.5

        accuracy=metrics.accuracy_score(targets,outputs)
        print("accuracy_score:",accuracy)

        if accuracy>best_accuracy:
            best_accuracy=accuracy
            torch.save(model.state_dict(),config.MODEL_PATH)
    
if __name__=="__main__":
    train()