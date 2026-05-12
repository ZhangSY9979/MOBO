import torch
from torchprofile import profile_macs
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time

from models import get_il_model
from utils.conf import set_random_seed
from argparse import Namespace
from models.utils.incremental_model import IncrementalModel
from datasets.utils.incremental_dataset import IncrementalDataset
from typing import Tuple
from datasets import get_dataset


def mask_classes(outputs: torch.Tensor, dataset: IncrementalDataset, k: int) -> None:
    cats = dataset.t_c_arr[k]
    outputs[:, 0:cats[0]] = -float('inf')
    outputs[:, cats[-1] + 1:] = -float('inf')


def evaluate(model: IncrementalModel, dataset: IncrementalDataset, last=False):
    accs_taskil, accs_classil, acc_taskil_pertask, acc_classil_pertask = [], [], [], []
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue

        correct_k, correct_mask_classes_k, total_k = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            correct_k += torch.sum(pred == labels).item()
            total_k += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
                correct_mask_classes_k += torch.sum(pred == labels).item()

        acc_classil_pertask.append(correct_k / total_k * 100)
        acc_taskil_pertask.append(correct_mask_classes_k / total_k * 100)

    accs_classil.append(correct / total * 100)
    accs_taskil.append(correct_mask_classes / total * 100)

    return accs_classil, accs_taskil, acc_classil_pertask, acc_taskil_pertask


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return round(np.mean(li), 2)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return round(np.mean(li), 2)


def train_il(args: Namespace) -> None:
    print(args)

    final_acc_class, final_acc_task = [], []
    final_bwt_class, final_bwt_task = [], []
    for run_id in range(args.repeat):
        print('================================= repeat {}/{} ================================='
              .format(run_id+1, args.repeat, ), file=sys.stderr)

        if args.seed is not None:
            set_random_seed(args.seed)
        if not os.path.exists(args.img_dir):
            os.makedirs(args.img_dir)

        dataset = get_dataset(args)
        model = get_il_model(args)

        nt = dataset.nt
        acc_track = []
        for i in range(nt):
            acc_track.append([0.0])

        model.begin_il(dataset)
        acc_classils, acc_taskils, acc_classils_pt, acc_taskils_pt = [], [], [], []  # pt = per_task
        for t in range(dataset.nt):

            train_loader, test_loader = dataset.get_data_loaders()

            start_time = time.time()
            model.train_task(dataset, train_loader)
            train_time = time.time() - start_time

            model.test_task(dataset, test_loader)

            start_time = time.time()
            accs_classil, accs_taskil, acc_classil_pertask, acc_taskil_pertask = evaluate(model, dataset)
            test_time = time.time() - start_time

            print('train_time', train_time, 'test_time', test_time)
            for i, acc in enumerate(acc_taskil_pertask):
                acc_track[i].append(acc)
                print('Acc for task', i, ':', acc_track[i])

            print_accuracy(accs_classil, accs_taskil, t + 1)
            acc_classils.append(accs_classil)
            acc_taskils.append(accs_taskil)
            acc_classils_pt.append(acc_classil_pertask)
            acc_taskils_pt.append(acc_taskil_pertask)


        # test_inference_delay(args, train_loader, model)

        model.end_il(dataset)

        for t in range(dataset.nt):
            accs_classil = acc_classils[t]
            accs_taskil = acc_taskils[t]
            print_accuracy(accs_classil, accs_taskil, t + 1)

        forget_taskil, forget_classil = forgetting(acc_taskils_pt), forgetting(acc_classils_pt)
        bwt_taskil, bwt_classil = backward_transfer(acc_taskils_pt), backward_transfer(acc_classils_pt)
        print('forget_taskil:', forget_taskil, 'forget_classil:', forget_classil, 'bwt_taskil:', bwt_taskil, 'bwt_classil:',
              bwt_classil, )

        torch.save(model, args.img_dir + '/' + args.dataset + '.pt')

        final_acc_class.append(np.mean(acc_classils[-1]))
        final_acc_task.append(np.mean(acc_taskils[-1]))
        final_bwt_class.append(bwt_classil)
        final_bwt_task.append(bwt_taskil)

    print('ACC for all {} task(s): \t [Class-IL]: {}$\\pm${} \t [Task-IL]: {}$\\pm${}'
          .format(args.repeat, round(np.mean(final_acc_class), 2), round(np.std(final_acc_class), 2), round(np.mean(final_acc_task), 2), round(np.std(final_acc_task), 2), file=sys.stderr))
    print('BWT for all {} task(s): \t [Class-IL]: {}$\\pm${} \t [Task-IL]: {}$\\pm${}'
          .format(args.repeat, round(np.mean(final_bwt_class), 2), round(np.std(final_bwt_class), 2), round(np.mean(final_bwt_task), 2), round(np.std(final_bwt_task), 2), file=sys.stderr))


def print_accuracy(accs_classil, accs_taskil, task_number: int) -> None:
    mean_acc_class_il = np.mean(accs_classil)
    mean_acc_task_il = np.mean(accs_taskil)
    print('Accuracy for {} task(s): \t [Class-IL]: {} %'
          ' \t [Task-IL]: {} %'.format(task_number, round(
        mean_acc_class_il, 2), round(mean_acc_task_il, 2), file=sys.stderr))


def test_inference_delay(args, dataloader, model):
    for sample in dataloader:
        break  # 只取第一个batch，然后停止
    sample = sample[0][:10, ...].to(args.device)
    print(sample.shape)

    # flops, params = profile_macs(model, sample)
    start_time = time.time()
    with torch.no_grad():
        if 'class-il' not in model.COMPATIBILITY:
            outputs = model(sample, 0)
        else:
            output = model(sample)
    end_time = time.time()

    print('Complex Information: flops:', 'params', 'delay:', end_time - start_time)