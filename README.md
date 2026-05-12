## MOBO: A Merging-Oriented Bi-Level Optimization Framework for Class Incremental Learning


## Abstract
Class-Incremental Learning (CIL) aims to enable models to sequentially learn new tasks while retaining knowledge from previous ones. Recently, merging-based pre-trained CIL methods have gained significant attention due to their competitive performance and high inference efficiency. However, most existing approaches decouple training from merging, neglecting the compatibility among task-specific adapters. This incompatibility introduces severe conflicts during integration, resulting in catastrophic forgetting and notable performance degradation. To address this limitation, we propose MOBO, a Merging-Oriented Bi-Level Optimization framework that synergizes the optimization of the current task model with the performance of the final merged model. At the upper level, we introduce a global loss that anticipates the merged model's behavior, guiding the current task parameters toward a solution space that facilitates effective merging. At the lower level, we employ a dynamic weighted merging strategy to optimize merging coefficients and update the merged model. By alternately optimizing the task-specific and merged models, MOBO effectively mitigates the performance loss caused by incompatible adapter integration. Comprehensive experiments on CIFAR-100, CUB-200, ImageNet-R, ImageNet-A, and VTAB demonstrate the superiority of our approach, highlighting its robustness and scalability, particularly on long task sequences.

### Set up
'''
python -u train_seed.py
'''
