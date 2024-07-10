# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# We changed it for adapted in our paper.
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
from sklearn.metrics import accuracy_score
from sklearn.metrics._ranking import roc_auc_score
import torch.nn.functional as F
import copy
from torchvision.utils import save_image


class DNELayer(torch.nn.Module):
    def __init__(self, shape):
        super(DNELayer, self).__init__()
        self.noise = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.noise

def train_dne(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        epoch: int,
        log_writer=None,
        args=None,
        dne_layer=None
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    assert dne_layer is not None, "Please pass a valida noisy layer"
    dne_layer.train()
    model.eval()
 
    optimizer_dne = torch.optim.Adam(dne_layer.parameters(), lr=5e-5)
    losses = []
    for images, labels, sa in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        sa = sa.to(device)

        optimizer_dne.zero_grad()
        adversarial_images = dne_layer(images)
        output_sa = model(adversarial_images)
        classification_loss = - args.lambda_1 * F.cross_entropy(output_sa, sa.view(-1))

        # Regularization term: L2 norm of the noise
        noise_l2_norm = torch.norm(dne_layer.module.noise, p=2)

        # Total loss = original loss + lambda * regularization term
        total_loss = classification_loss + args.lambda_reg * noise_l2_norm

        total_loss.backward()
        optimizer_dne.step()
        losses.append(total_loss.item())
    print("DNE Loss: ", np.array(losses).mean())
    with torch.no_grad():
        print("Magnitute of noise: ", dne_layer.module.noise.mean())


def finetune_fm(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, last_activation=None, dne_layer=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    model.train()
    for data_iter_step, (samples, targets, sa) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        if dne_layer is not None:
            samples = dne_layer(samples)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if last_activation is not None:
            if last_activation == 'sigmoid':
                last_activation = torch.nn.Sigmoid()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if last_activation is not None:
                outputs = last_activation(outputs)

            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_dne_gezo(      
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        epoch: int,
        log_writer=None,
        args=None,
        dne_layer=None
    ):
    original_dne = dne_layer.module.noise.data.clone()
    best_direction = None
    best_iteration_loss = float('inf')
    velocity = torch.zeros_like(dne_layer.module.noise.data)
    step_size = args.init_step

    # Sample multiple perturbations
    for _ in range(args.sampled_steps):  # Increase number of samples
        perturbation = torch.randn_like(dne_layer.module.noise) * step_size
        for i, (images, _, sa) in enumerate(data_loader):
            images = images.to(device)
            sa = sa.to(device)
            
            for sign in [-1, 1]:  # Explore both directions
                direction_noise = original_dne + sign * perturbation
                dne_layer.module.noise.data = direction_noise
                iteration_losses = []
            

                with torch.no_grad():
                    adversarial_images = dne_layer(images)
                    output_sa = model(adversarial_images)
                    loss = - args.lambda_1 * F.cross_entropy(output_sa, sa.view(-1)) + args.lambda_reg * torch.norm(dne_layer.module.noise, p=2)
                    iteration_losses.append(loss.item())

            mean_loss = np.mean(iteration_losses)
            if mean_loss < best_iteration_loss:
                best_iteration_loss = mean_loss
                best_direction = sign * perturbation

            if i >= 1:
                break

    if best_direction is not None:
        velocity = args.momentum * velocity + best_direction  # Update velocity with momentum
        dne_layer.module.noise.data = original_dne + velocity
        best_loss = best_iteration_loss
    else:
        step_size *= 0.95  # Reduce step size if no improvement

    print(f"Best Loss = {best_loss}, Step Size = {step_size}")
    with torch.no_grad():
        print("Magnitute of noise: ", dne_layer.module.noise.mean())
    



@torch.no_grad()
def evaluate_disease(data_loader, model, device, args, dne_layer=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target, _ = batch[0], batch[1], batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if dne_layer is not None:
                images = dne_layer(images)
            output = model(images)
            loss = criterion(output, target)

        acc1 = accuracy_multihot(output, target, topk=(1, ))[0]
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        outputs.append(output)
        targets.append(target)
    
    targets = torch.cat(targets, dim=0).cpu().numpy()
    outputs = torch.cat(outputs, dim=0).sigmoid().cpu().numpy()
    auc_each_class = computeAUROC(targets, outputs, args.nb_classes)
    auc_each_class_array = np.array(auc_each_class)
    missing_classes_index = np.where(auc_each_class_array == 0)[0]
    if missing_classes_index.shape[0] > 0:
        print('There are classes that not be predicted during testing,'
              ' the indexes are:', missing_classes_index)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()}}

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    # print(dataGT.shape, dataPRED.shape)
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    print(outAUROC)
    return outAUROC


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def accuracy_multihot(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # Assuming output is [batch_size, num_classes] and target is [batch_size, 2] in one-hot format
    # Convert target from one-hot encoding to class indices
    _, target_indices = target.max(dim=1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    # Compare with target indices instead of one-hot encoded target
    correct = pred.eq(target_indices.view(1, -1).expand_as(pred))

    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


@torch.no_grad()
def evaluate_sa(data_loader, model, device, args):

    if args.dataset == 'chexpert':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_outputs = []
    all_sas = []
    all_preds = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, _, sa = batch

        images = images.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():              
            output = model(images)
            loss = criterion(output, sa.view(-1))

        if args.dataset == "chexpert":
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
        else:
            NotImplementedError()

        all_outputs.extend(output.cpu().numpy())
        all_sas.extend(sa.cpu().numpy())


        metric_logger.update(loss=loss.item())

    num_classes = args.nb_classes


    accuracy = accuracy_score(all_sas, all_preds)
    auc_each_class = computeAUROC(all_sas, all_outputs, num_classes)
    auc_each_class_array = np.array(auc_each_class)

    auc_avg = np.average(auc_each_class_array[auc_each_class_array != 0])
    metric_logger.synchronize_between_processes()

    print('Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {**{'acc': accuracy, 'auc_avg': auc_avg, 'auc_each_class': auc_each_class}}