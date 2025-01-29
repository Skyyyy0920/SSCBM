import os
import re
import glob
import copy
import torch
import logging
import itertools
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt

import data.pt_loader as pt_data_module
import data.pbc_loader as pbc_data_module
import data.cub_loader as cub_data_module
import data.awa2_loader as awa_data_module
import data.celeba_loader as celeba_data_module


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
            _, y, _, _, _, _ = data
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


def generate_dataset_and_update_config(experiment_config, args):
    dataset_config = experiment_config['dataset_config']
    if args.dataset == "CUB-200-2011":
        data_module = cub_data_module
    elif args.dataset == "PBC":
        data_module = pbc_data_module
    elif args.dataset == "7pt":
        data_module = pt_data_module
    elif args.dataset == "CelebA":
        data_module = celeba_data_module
    elif args.dataset == "AwA2":
        data_module = awa_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = data_module.generate_data(
        config=dataset_config,
        seed=args.seed,
        labeled_ratio=args.labeled_ratio,
    )
    logging.info(f"imbalance: {imbalance}")

    intervention_config = experiment_config.get('intervention_config', {})
    acquisition_costs = None
    if concept_map is not None:
        intervened_groups = list(range(0, len(concept_map) + 1, intervention_config.get('intervention_freq', 1)))
    else:
        intervened_groups = list(range(0, n_concepts + 1, intervention_config.get('intervention_freq', 1)))

    task_class_weights = update_config_with_dataset(
        config=experiment_config,
        train_dl=train_dl,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_map=concept_map,
    )

    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
        task_class_weights,
        acquisition_costs,
    )


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


def visualize_and_save_heatmaps(
        x,
        c_ground_truth,
        c_pred,
        heatmap,
        output_dir='output_images',
        concept_set=None,
):
    """
    x (torch.Tensor): [channels, height, width]
    heatmap (torch.Tensor): [num_heatmaps, heatmap_height, heatmap_width]
    """
    os.makedirs(output_dir, exist_ok=True)

    image = x.permute(1, 2, 0).numpy()  # 转换成 (height, width, channels)

    # 保存原图
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'original_image.png'))
    plt.close()

    # 生成红色蒙版
    red_mask = np.zeros_like(image)
    red_mask[..., 0] = 1.0  # 红色通道
    red_mask[..., 1] = 0.0  # 绿色通道
    red_mask[..., 2] = 0.0  # 蓝色通道

    # 对每个热图进行处理并保存
    for i in range(heatmap.shape[0]):
        # 取出第 i 个通道的 heatmap
        hm = heatmap[i].unsqueeze(0).unsqueeze(0)  # (1, 1, heatmap_height, heatmap_width)

        # 上采样到图像大小
        hm_upsampled = F.interpolate(hm, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
        hm_upsampled = hm_upsampled.squeeze().numpy()  # (height, width)

        # 根据 heatmap 的值设置透明度
        alpha = hm_upsampled / hm_upsampled.max()  # 归一化到 0-1
        alpha = np.clip(alpha, 0, 1)  # 保证值在 0-1 之间

        # 叠加蒙版到原始图像上
        overlay = image.copy()
        for chn in range(3):
            overlay[..., chn] = image[..., chn] * (1 - alpha) + red_mask[..., chn] * alpha

        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f'{concept_set[i]}    [{c_ground_truth[i]}] [{c_pred[i]:.3f}]')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{concept_set[i]}'))
        plt.close()
