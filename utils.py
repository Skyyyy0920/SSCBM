import os
import re
import glob
import copy
import torch
import logging
import itertools
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def zipdir(path, zipf, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                zipf.write(filename, arcname)


def logging_config(save_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'running.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def update_config_with_dataset(
        config,
        train_dl,
        n_concepts,
        n_tasks,
        concept_map,
):
    config["n_concepts"] = n_concepts
    config["n_tasks"] = n_tasks
    config["concept_map"] = concept_map

    task_class_weights = None

    if config.get('use_task_class_weights', False):
        logging.info(f"Computing task class weights in the training dataset with size {len(train_dl)}...")
        attribute_count = np.zeros((max(n_tasks, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            _, y, _, _ = data
            if n_tasks > 1:
                y = torch.nn.functional.one_hot(y, num_classes=n_tasks).cpu().detach().numpy()
            else:
                y = torch.cat(
                    [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                    dim=-1).cpu().detach().numpy()
            attribute_count += np.sum(y, axis=0)
            samples_seen += y.shape[0]
        logging.info(f"Class distribution is: {attribute_count / samples_seen}")
        if n_tasks > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array([attribute_count[0] / attribute_count[1]])

    return task_class_weights


def generate_hyper_param_configs(config):
    if "grid_variables" not in config:
        # Then nothing to see here, so we will return a singleton set with this config in it
        return [config]
    # Else time to do some hyperparameter search in here!
    vars = config["grid_variables"]
    options = []
    for var in vars:
        if var not in config:
            raise ValueError(f'All variable names in "grid_variables" must be existing '
                             f'fields in the config. However, we could not find any field with name "{var}".')
        if not isinstance(config[var], list):
            raise ValueError(f'If we are doing a hyper-paramter search over variable "{var}", '
                             f'we expect it to be a list of values. Instead we got {config[var]}.')
        options.append(config[var])
    mode = config.get('grid_search_mode', "exhaustive").lower().strip()
    if mode in ["grid", "exhaustive"]:
        iterator = itertools.product(*options)
    elif mode in ["paired"]:
        iterator = zip(*options)
    else:
        raise ValueError(f'The only supported values for grid_search_mode '
                         f'are "paired" and "exhaustive". We got {mode} instead.')
    result = []
    for specific_vals in iterator:
        current = copy.deepcopy(config)
        for var_name, new_val in zip(vars, specific_vals):
            current[var_name] = new_val
        result.append(current)
    return result


def evaluate_expressions(config, parent_config=None, soft=False):
    parent_config = parent_config or config
    for key, val in config.items():
        if isinstance(val, (str,)):
            if len(val) >= 4 and val[0:2] == "{{" and val[-2:] == "}}":
                # Then do a simple substitution here
                try:
                    config[key] = val[2:-2].format(**parent_config)
                    config[key] = eval(config[key])
                except Exception as e:
                    if soft:
                        # Then we silently ignore this error
                        pass
                    else:
                        # otherwise we just simply raise it again!
                        raise e
            else:
                config[key] = val.format(**parent_config)
        elif isinstance(val, dict):
            # Then we progress recursively
            evaluate_expressions(val, parent_config=parent_config)
