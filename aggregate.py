import torch
import numpy as np
from metric.iou import IoU

def MC_prediction(model, img, n_MC=24, softmax_temperature=1.):
    """
    Run several inferences with MC dropout and return the prediction
    (highest average softmax value), the corresponding average softmax value
    and the variance of softmax value
    """
    with torch.no_grad():
        tmp = (model(img)/softmax_temperature).softmax(dim=1).detach().cpu()
        pred_MC = torch.zeros((n_MC, *tmp.shape), dtype=torch.float32)
        pred_MC[0] = tmp
        for ii in range(1,n_MC):
            pred_MC[ii] = model(img).softmax(dim=1).detach().cpu()
        mean_MC = pred_MC.mean(dim=0)
        var_MC = pred_MC.var(dim=0)
        mean_confidence, prediction = mean_MC.max(dim=1)
        variance = var_MC.gather(1, prediction.unsqueeze(1)).squeeze(1)
    return prediction, mean_confidence, variance

def MC_prediction_counts(model, img, n_MC=24):
    """
    Run several inferences with MC dropout and return the most frequent prediction
    and the proportion of times it was predicted
    """
    with torch.no_grad():
        tmp = model(img).detach().cpu()
        pred_MC = torch.zeros((n_MC, *tmp.shape), dtype=torch.float32)
        pred_MC[0] = tmp
        for ii in range(1,n_MC):
            pred_MC[ii] = model(img).detach().cpu()
            
        count_MC = torch.zeros(pred_MC.shape[1:], dtype=torch.long)
        _, label_MC = torch.max(pred_MC,dim=2)
        for label in range(count_MC.size(1)):
            count_MC[:,label,:,:] = label_MC.eq(label).sum(dim=0)
            
        conf, prediction = count_MC.max(dim=1)
        conf = conf.float()/n_MC
            
    return prediction, conf

def evaluate_model_MCdrop(model, loader, n_gt, n_pred, bins,
                          device=None,
                          softmax_temperature = 1.0,
                          kappa = -1.5, n_MC=24,
                          method="mean"):
    """
    Input:
        model: torch model
        loader: dataloader that yields (image, label) tensors
        n_gt: number of ground-truth classes (in the label tensor)
        n_pred: number of predicted classes (number of channels of model output)
        bins: bin edges for histogram calculation
        device: torch.device, where the model is located
        softmax_temperature: temperature for the softmax
        kappa: adjusment variable for confidence calculation. The confidence used is
               (mean_msv - kappa*sqrt(var_msv)) whare mean_msv and var_msv are the
               mean and variance of max softmax value
        n_MC: number of repetitions for Monte Carlo estimate
        method: "mean" for highest average of soft prediction
                "count" for mode of hard predictions
    
    Returns:
        hist_array : 3D array, hist_array[true_label, predicted_label] is the histogram
                    of prediciton confidence for this (true, predicted) pair
        iou : IoU metric per class
        miou : IoU average
    """
    if device==None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hist_array = np.zeros((n_gt,n_pred,len(bins)-1), dtype=int)
    iou_metric = IoU(12)
    for img, gt_label in loader:
        
        if method=="mean":
            predicted_label, mean_confidence, variance = MC_prediction(model,img.to(device),n_MC,softmax_temperature)
            confidence = torch.clamp(mean_confidence + kappa*variance,0.,1.) 
        elif method=="count":
            predicted_label, confidence = MC_prediction_counts(model,img.to(device),n_MC)
        else:
            raise ValueError("Method: expected 'mean' or 'count'")
            
        predicted_label, confidence = map(lambda t:t.detach().cpu(),
                                          (predicted_label, confidence))
        
        iou_metric.add(predicted_label,gt_label)
        
        predicted_label, confidence, gt_label = map(lambda t:t.numpy(),
                                                    (predicted_label, confidence, gt_label))
        
        for ii in range(n_pred):
            sel = (predicted_label == ii)
            conf_, gt_ = confidence[sel], gt_label[sel]
            
            for jj in range(n_gt):
                hist_array[jj,ii] += np.histogram(conf_[gt_ == jj], bins)[0]
    
    iou, miou = iou_metric.value()
    return (hist_array, iou, miou)


def aggregate_prediction(models, img, softmax_temperature = 1.):
    """
    Run several inferences with a list of models and return the prediction
    (highest average softmax value), the corresponding average softmax value
    and the variance of softmax value
    """
    with torch.no_grad():
        tmp = (models[0](img) / softmax_temperature).softmax(dim=1).detach().cpu()
        pred_MC = torch.zeros((len(models), *tmp.shape), dtype=torch.float32)
        pred_MC[0] = tmp
        for ii, model in enumerate(models[1:]):
            pred_MC[ii+1] = model(img).softmax(dim=1).detach().cpu()
        mean_MC = pred_MC.mean(dim=0)
        var_MC = pred_MC.var(dim=0)
        
        mean_confidence, prediction = mean_MC.max(dim=1)
        variance = var_MC.gather(1, prediction.unsqueeze(1)).squeeze(1)
    return prediction, mean_confidence, variance

def evaluate_model_aggregate(models, loader, n_gt, n_pred, bins,
                          device=None,
                          softmax_temperature = 1.0,
                          kappa = -1.5):
    """
    Input:
        models: list of torch models
        loader: dataloader that yields (image, label) tensors
        n_gt: number of ground-truth classes (in the label tensor)
        n_pred: number of predicted classes (number of channels of model output)
        bins: bin edges for histogram calculation
        device: torch.device, where the model is located
        softmax_temperature: temperature for the softmax
        kappa: adjusment variable for confidence calculation. The confidence used is
               (mean_msv - kappa*sqrt(var_msv)) whare mean_msv and var_msv are the
               mean and variance of max softmax value
        n_MC: number of repetitions for Monte Carlo estimate
    
    Returns:
        hist_array : 3D array, hist_array[true_label, predicted_label] is the histogram
                    of prediciton confidence for this (true, predicted) pair
        iou : IoU metric per class
        miou : IoU average
    """
    if device==None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hist_array = np.zeros((n_gt,n_pred,len(bins)-1), dtype=int)
    iou_metric = IoU(12)
    for img, gt_label in loader:
        
        predicted_label, mean_confidence, variance = aggregate_prediction(models,
                                                                          img.to(device),
                                                                          softmax_temperature)
        confidence = torch.clamp(mean_confidence + kappa*variance,0.,1.) 
        
        predicted_label, confidence = map(lambda t:t.detach().cpu(),
                                          (predicted_label, confidence))
        
        iou_metric.add(predicted_label,gt_label)
        
        predicted_label, confidence, gt_label = map(lambda t:t.numpy(),
                                                    (predicted_label, confidence, gt_label))
        
        for ii in range(n_pred):
            sel = (predicted_label == ii)
            conf_, gt_ = confidence[sel], gt_label[sel]
            
            for jj in range(n_gt):
                hist_array[jj,ii] += np.histogram(conf_[gt_ == jj], bins)[0]
    
    iou, miou = iou_metric.value()
    return (hist_array, iou, miou)