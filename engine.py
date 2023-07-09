#engine.py
import config
import transformers
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

def loss_fn(output,targets):
    """
    param outputs: output from the model (real numbers)
    param targets: input targets
    """
    return nn.BCEWithLogitsLoss()(output,targets.view(-1,1))

def train_fn(data_loader,model,optimizer,device,scheduler):
    """
    this is the training function
    :param dataloader: dataloader object
    :param model: torch model
    :param optimizer: adam,sgd,etc
    :param deevice: gpu/cpu
    :param scheduler: learning rate scheduler
    """

    #put the model in train mode
    model.train()

    #loop over all the batches
    for d in data_loader:
        #extract ids,token_id and the mask and the targets
        ids=d["ids"]
        token_type_ids=d["token_type_ids"]
        mask=d["mask"]
        targets=d["targets"]

        #move everything to the specified device
        ids=ids.to(device,dtype=torch.long)
        token_type_ids=token_type_ids.to(device,dtype=torch.long)
        mask=mask.to(device,dtype=torch.long)
        targets=targets.to(device,dtype=torch.float)

        ## zero grad the optimizer
        optimizer.zero_grad()

        #pass through the model
        output=model(ids=ids,mask=mask,token_type_ids=token_type_ids)
        #calculate the loss of the model
        loss=loss_fn(output,targets)
        loss.backward()

        optimizer.step()
        scheduler.step()


def eval_fn(data_loader,model,device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]

    with torch.no_grad():
        for d in data_loader:
            ids=d["input_ids"]
            mask=d["mask"]
            token_type_ids=d["token_type_ids"]
            target=d["targets"]

            #move everything to device
            ids=ids.to(device,dtype=torch.long)
            mask=mask.to(device,dtype=torch.long)
            token_type_ids=token_type_ids.to(device,dtype=torch.long)
            target=target.to(device,dtype=torch.float)

            outputs=model(ids,mask,token_type_ids)
            target=target.cpu().detach()
            fin_targets.extend(target.numpy().tolist())

            torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())
    return fin_outputs,fin_targets





