import random
import numpy as np
import torch

def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))

    
def get_model_params_num(model):
    return sum(np.product(t.shape) for t in model.parameters())


def create_model_name(**args):
    model_name = []
    for arg_name, arg_value in args.items():
        model_name.append(arg_name)
        model_name.append(str(arg_value))
    return "_".join(model_name)

def parse_model_name(model_name):
    model_params = {}
    model_name = model_name.split("_")
    for i in range(len(model_name) // 2):
        param_name = model_name[2*i]
        param_value = model_name[2*i+1]
        model_params[param_name] = param_value
    return model_params