import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


# train 함수 for one epoch
def train(device, dataloader:DataLoader, model:nn.Module, loss_function, optimizer:torch.optim.Optimizer):
    model.train() # train 모드 진입
    total_loss = 0. # initialize 
    for X, y in dataloader: # dataloader에서 알아서 batch별로 꺼내올거임
        X, y = X.to(device), y.to(device)
        # compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss 직접 계산
        total_loss += loss.item()*len(y)  # loss.item() loss for every batch 
    return total_loss/len(dataloader.dataset) # 평균 loss값

# test 셋에 넣어서 바로바로 metric 확인해볼 용도, test셋에서도 학습 잘 되는지 확인부터 해야하니까
def test(device, dataloader:DataLoader, model:nn.Module, loss_function, metric:torchmetrics.metric.Metric= None, return_pred:bool=False):  # return_pred이 True일때만, prediction값 return
    model.eval() # eval 모드 진입
    with torch.inference_mode():   
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # compute prediction error
            pred = model(X)
            tst_loss = loss_function(pred, y).item()
            accuracy = BinaryAccuracy(threshold=0.5)(pred, y).item()
            if metric is not None:
                metric.update(pred, y)
            pred_np = pred.numpy()  ### prediction 값 내보내기 위함
    if return_pred:
        return tst_loss, accuracy, pred_np
    return tst_loss, accuracy
        