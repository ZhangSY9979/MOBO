import copy
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from backbone.linears import CosineLinear
from backbone.vit_acmap import vit_base_patch16_224, vit_base_patch16_224_in21k
from utils.context import Context
from utils.merge import initial_merge, merge


class MOBONet(nn.Module):
    def __init__(self, context: Context):
        super().__init__()
        self.cov_inv = None
        self.c = context
        self.config = self.c.config
        self.logger = self.c.logger
        self.out_dim = self.config.transformer.out_dim
        self.use_init_ptm = self.config.exp.use_init_ptm
        self.device = self.config.device

        self.backbone = self.get_backbone()

        self.fc: Optional[CosineLinear] = None
        self.merged_adapter_list: List[Optional[nn.ModuleList]] = []
        self.old_adapter_list: List[Optional[nn.ModuleList]] = []
        self.protos_list_source = []
        self.proxy_fc_list = []
        self.task_proto = []
        self.task_cov = []
        self.old_num = 0

        self.old_fisher = None
        self.new_fisher = None

    def get_backbone(self):
        backbone_type = self.config.exp.backbone_type.lower()

        self.logger.info('Loading the pretrained model from timm...')
        if backbone_type == 'vit_base_patch16_224':
            model = vit_base_patch16_224(config=self.config)
        elif backbone_type == 'vit_base_patch16_224_in21k':
            model = vit_base_patch16_224_in21k(config=self.config)
        else:
            raise NotImplementedError(f'Unknown type {backbone_type}')

        return model

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def copy(self):
        return copy.deepcopy(self)

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * 2
        else:
            return self.out_dim * 1

    def update_proxy_fc(self):
        self.proxy_fc = self.generate_fc(self.out_dim, self.c.cur_task_size).to(self.device)

    def update_fc(self):
        new_fc = self.generate_fc(self.feature_dim, self.c.total_classes).to(self.device)
        new_fc.reset_parameters_to_zero()

        if self.fc is None:
            self.fc = new_fc
            return

        old_nb_classes = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        new_fc.sigma.data = self.fc.sigma.data
        if new_fc.weight.shape[1] != weight.shape[1]:
            new_fc.weight.data[:old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        else:
            new_fc.weight.data[:old_nb_classes, :] = nn.Parameter(weight)

        self.fc = new_fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def replace_fc(self, dataset, data_loader):
        assert self.fc is not None

        ptm_index = 0
        protos_current = self.extract_prototype(dataset, data_loader, self.merged_adapter_list[-1])


        if not self.c.is_first_task:
            old_protos = self.extract_prototype(dataset, data_loader, self.merged_adapter_list[-2])

            self.task_proto = []
            protos_list_mapped = []
            for i, protos in enumerate(self.protos_list_source):  # len = T C*L
                proto_shift = torch.mean(protos_current, dim=0).to(self.device) - torch.mean(old_protos, dim=0).to(
                    self.device
                )

                self.protos_list_source[i] = protos.to(self.device) + proto_shift.to(self.device)  # P_i(A_i) + P_t(A_t) - P_t(A_i)

                self.task_proto.append(self.protos_list_source[i])
                protos_list_mapped.append(self.protos_list_source[i])
        self.task_proto.append(protos_current)
        self.protos_list_source.append(protos_current)

        if not self.c.is_first_task:
            fc_mapped = torch.cat(protos_list_mapped, dim=0)

            self.fc.weight.data[: -protos_current.shape[0],
            self.out_dim * ptm_index: self.out_dim * (ptm_index + 1)] = (
                fc_mapped
            )
        self.fc.weight.data[-protos_current.shape[0]:, self.out_dim * ptm_index: self.out_dim * (ptm_index + 1)] = (
            protos_current
        )


    def extract_prototype(self, dataset, data_loader, adapter):
        with torch.no_grad():
            prog_bar = data_loader

            # extract embeddings
            embedding_list, label_list = [], []
            for _, batch in enumerate(prog_bar):
                (_, data, label) = batch
                data = data.to(self.device)
                label = label.to(self.device)
                embedding = self.backbone.forward_proto(data, adapter)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())


            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            # construct prototype-based classifier
            class_list = np.unique(dataset.labels)
            proto_list = []
            for class_index in class_list:
                # calculate prototype
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                proto_list.append(proto[None])

            return torch.cat(proto_list, dim=0)

    def update_cov(self, dataset, data_loader, adapter):
        with torch.no_grad():
            prog_bar = data_loader

            embedding_list, label_list = [], []
            for _, batch in enumerate(prog_bar):
                (_, data, label) = batch
                data = data.to(self.device)
                label = label.to(self.device)
                embedding = self.backbone.forward_proto(data, adapter)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())


            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            class_list = np.unique(dataset.labels)
            proto_list = []
            for class_index in class_list:
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                proto_list.append(proto[None])

            eps = 1e-6

            D = embedding_list.size(1)
            shared_cov = torch.zeros(D, D)
            for c in class_list:
                data_index = (label_list == c).nonzero().squeeze(-1)
                X_c = embedding_list[data_index]
                mu_c = X_c.mean(dim=0, keepdim=True)
                diff = X_c - mu_c
                shared_cov += diff.T @ diff
            shared_cov /= (embedding_list.size(0) - len(class_list))

            shrinkage = 0.1
            trace_avg = torch.trace(shared_cov) / D
            shared_cov = (1 - shrinkage) * shared_cov + shrinkage * trace_avg * torch.eye(D)
            shared_cov += eps * torch.eye(D)
            cov_result = shared_cov

            if len(self.task_cov) > 0:
                # 协方差更新
                old_cov = self.task_cov[0]
                new_cov = cov_result
                old_num = self.old_num
                new_num = embedding_list.shape[0]
                ans_cov = (old_num * old_cov + new_num * new_cov) / (old_num + new_num)
                self.task_cov[0] = ans_cov
                self.old_num = (old_num + new_num)
            else:
                self.task_cov.append(cov_result)
                self.old_num = embedding_list.shape[0]

            cov = self.task_cov[0].to(self.device)
            self.cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0]).to(self.device))  # 加小正则项避免奇异
            return torch.cat(proto_list, dim=0)


    def alpha(self):
        with torch.no_grad():
            value_old = 0.0
            value_new = 0.0

            for (n_cur, p_cur), (n_prev, p_prev) in zip(
                    self.backbone.cur_adapter.named_parameters(),
                    self.merged_adapter_list[-1].named_parameters(),
            ):

                F_diag_old = self.old_fisher[n_cur]
                F_diag_new = self.new_fisher[n_cur]


                diff = p_cur - p_prev.detach()


                value_old += (F_diag_old * diff.pow(2)).sum().item()
                value_new += (F_diag_new * diff.pow(2)).sum().item()
            margin_alpha = value_old / (value_new + value_old)
            final_margin_alpha = margin_alpha

        return final_margin_alpha

    def merge_adapters(self):
        if len(self.merged_adapter_list) >= self.config.our.limit_centroid_map:
            self.logger.info('Skip merge')
            return

        merge_method = self.config.our.merge_method
        self.logger.info(f'Merge method: {merge_method}')

        # not merge
        if merge_method == 'simple':
            self.merged_adapter_list.append(None)
            return

        # merge adapters
        if self.c.is_first_task:
            # initialize merged adpater
            merged_state_dict = initial_merge(
                config=self.config,
                merge_method=merge_method,
                cur_adapter=self.backbone.cur_adapter,
            )
        else:
            # obtain merged adpater
            assert self.merged_adapter_list[-1] is not None

            final_margin_alpha = self.alpha()
            merged_state_dict = merge(
                config=self.config,
                merge_method=merge_method,
                prev_adapter=self.merged_adapter_list[-1],
                cur_adapter=self.backbone.cur_adapter,
                num_adapters=final_margin_alpha,
            )
            for k in self.old_fisher:
                self.old_fisher[k] += self.new_fisher[k]

        with torch.no_grad():
            merged_adapter = copy.deepcopy(self.backbone.cur_adapter).requires_grad_(False)
            merged_adapter.load_state_dict(merged_state_dict)
            self.merged_adapter_list.append(merged_adapter)

    def extract_feature(self, x, adapter):
        # assert self.fc is not None
        adapter_list = [adapter]  # the last adapter
        return self.backbone.forward_feature(x, adapter_list)

    def mahalanobis_classifier(self, x):
        device = x.device
        logits_all = []
        cov_inv = self.cov_inv.to(self.device)

        for task_idx, proto in enumerate(self.task_proto):
            proto = proto.to(device)
            diff = x.unsqueeze(1) - proto.unsqueeze(0)  # [B, 5, D]
            dist = torch.einsum('bcd,de,bce->bc', diff, cov_inv, diff)  # [B, 5]

            task_logits = -dist
            logits_all.append(task_logits)

        logits = torch.cat(logits_all, dim=1)
        return {"logits": logits}

    def forward(self, x, test=False):
        if not test:
            # Training
            x = self.backbone.forward_train(x)
            out = self.proxy_fc(x)
        else:
            # Testing
            assert self.fc is not None
            adapter_list = [self.merged_adapter_list[-1]]
            if self.use_init_ptm:
                adapter_list = adapter_list + [None]
            x = self.backbone.forward_test(x, adapter_list)

            out = self.mahalanobis_classifier(x)
        out.update({'features': x})
        return out
