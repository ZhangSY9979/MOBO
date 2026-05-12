from utils.context import Context
import importlib

def get_model(context: Context):
    model_name = context.config.exp.name
    name = model_name
    module_name = f'models.{name}'

    try:
        module = importlib.import_module(module_name)
        return module.Learner(context=context)
    except ImportError:
        raise NotImplementedError(f'Model {name} is not implemented.')


