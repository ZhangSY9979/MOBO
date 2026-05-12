import yaml
import sys
from utils import factory
from utils.config import Config
from utils.context import Context
from utils.data_manager import DataManager
from utils.logger import BasicLogger, WandbLogger
from utils.options import setup_parser
from utils.toolkit import count_parameters, set_random, print_information
import os

def main():
    args = setup_parser().parse_args()
    args.config = './exps/vtab.yaml'
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config.update(vars(args))
    config.update({'seed': 1993})

    config = Config(**config)
    config = Config.model_validate(config)
    

    train(config=config, seed=config.seed)



def train(config: Config, seed):
    # setup device
    set_random(seed)

    # setup datasets
    data_manager = DataManager(
        dataset_name=config.exp.dataset, shuffle=config.exp.shuffle, seed=seed, dataset_dir=config.dataset_dir
    )

    # setup logger
    make_ckpts = not config.debug
    if config.logger == 'wandb':
        logger = WandbLogger(config, seed=seed, make_ckpts=make_ckpts, project_name='cil-acmap')
    elif config.logger == 'basic':
        logger = BasicLogger(config, seed=seed, make_ckpts=make_ckpts)
    else:
        raise ValueError('Invalid logger type.')

    context = Context(config=config, logger=logger, class_order=data_manager.class_order)

    # setup model
    model = factory.get_model(context=context)
    logger.info(f'All params: {count_parameters(model.network)}')
    logger.print_args()

    # training
    cnn_curve = {'top1': [], 'top5': []}
    for task in range(1, context.num_tasks + 1):
        logger.info(f'Task {task}/{context.num_tasks} ========================================================')

        # train
        model.incremental_train(data_manager=data_manager)

        # inference
        cnn_accy, _ = model.eval_task()

        if not config.debug:
            model.save_checkpoint(logger.ckpts_dir)

        # after task
        model.after_task()

        print_information(model, task, context.num_tasks)



if __name__ == '__main__':
    main()
