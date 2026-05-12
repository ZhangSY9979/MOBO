import copy
import json
import os
import random

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import sys
from backbone.vit_acmap import Adapter
from utils.config import Config

import torch.nn.functional as F

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy  each_task_class_ans
    task_id = -1
    for class_id in range(0, np.max(y_true), increment):
        task_id += 1
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id + increment - 1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
        all_acc[str(task_id)] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)


    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc['old'] = (
        0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{device}')

        gpus.append(device)

    args['device'] = gpus


def get_device_name(device_list):
    device_name = []
    for device in device_list:
        if device == -1:
            device_name.append('cpu')
        else:
            device_name.append(f'cuda:{device}')

    return device_name


def set_random(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def load_adapters(config: Config, ckpts_dir, task_indexes):
    _adapter = nn.ModuleList()
    for _ in range(config.transformer.depth):
        _adapter.append(Adapter(config=config))

    adapter_list = []
    for task in task_indexes:
        ckpt_model = torch.load(os.path.join(ckpts_dir, f'task{task}.pkl'))

        adapter = copy.deepcopy(_adapter)
        adapter.load_state_dict(ckpt_model['state_dict'])
        adapter_list.append(adapter)

    init_adapter = copy.deepcopy(_adapter)
    ckpt_model = torch.load(os.path.join(ckpts_dir, 'init.pkl'))
    init_adapter.load_state_dict(ckpt_model['state_dict'])

    return adapter_list, init_adapter


def print_information(model, task, num_tasks):
    print("****************Task ACC********************")
    for i in range(len(model.each_task)):
        print(f"Task ACC for task{i + 1}: {model.each_task[i]}")
    print("****************Class ACC********************")
    for i in range(len(model.each_class)):
        print(f"Class ACC for task{i + 1}: {model.each_class[i]}")

    print("****************Final ACC********************")
    if task == num_tasks:
        for i in range(len(model.total_result)):
            print(
                'Accuracy for {:2d} task(s): \t [Task-id]: {:6.2f} % \t [Task-IL]: {:6.2f} %  \t [Class-IL]: {:6.2f} %  \t [Average]: {:6.2f} %'.format(
                    i,
                    model.total_result[i][0],
                    model.total_result[i][1],
                    model.total_result[i][2],
                    model.total_result[i][3]
                ),
                file=sys.stderr
            )
    else:
        print(
            'Accuracy for {:2d} task(s): \t [Task-id]: {:6.2f} % \t [Task-IL]: {:6.2f} %  \t [Class-IL]: {:6.2f} %  \t [Average]: {:6.2f} %'.format(
                task,
                model.total_result[-1][0],
                model.total_result[-1][1],
                model.total_result[-1][2],
                model.total_result[-1][3]
            ),
            file=sys.stderr
        )
    print(" ")