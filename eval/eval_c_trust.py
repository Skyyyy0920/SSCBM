import os
import yaml
import torch
import logging
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from gradcam import CBMGradCAM, CBMActivationMapGenerator
from local_parts import *
from collections import defaultdict
from utils import *
from train.evaluate import *
from configs.basic_config import *
from models.construction import construct_model
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from data.cub_loader import CONCEPT_SEMANTICS, SELECTED_CONCEPTS, CUBDataset_for_heatmap

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def evaluate_concept_trustworthiness(all_activation_maps, all_img_ids, bbox_half_size=36, img_size=224):
    """
    all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute
    all_img_ids[i] : img_id for the i-th image
    """
    # 获取设备信息
    device = all_img_ids.device

    # Get the ground-truth attributes of the all test images
    all_attributes = []
    for img_id in all_img_ids:
        attributes = id_to_attributes[img_id.item()]
        all_attributes.append(attributes)
    all_attributes = np.stack(all_attributes, axis=0)
    all_attributes = torch.from_numpy(all_attributes).to(device)
    n_attributes = all_attributes.shape[-1]

    # Gather the part locs of each image
    all_img_num, part_num = all_img_ids.shape[0], 15
    all_part_locs = np.zeros((all_img_num, part_num, 2)) - 1  # Each element is (gt_y, gt_x)
    all_img_ids_cpu = all_img_ids.cpu().numpy()
    for idx, img_id in enumerate(all_img_ids_cpu):
        img_id = img_id.item()
        bbox = id_to_bbox[img_id]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        part_locs = id_to_part_loc[img_id]
        for part_loc in part_locs:
            part_id = part_loc[0] - 1  # Begin From 0
            loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
            ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (bbox_y2 - bbox_y1)
            re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
            all_part_locs[idx, part_id, 0] = re_loc_y
            all_part_locs[idx, part_id, 1] = re_loc_x
    all_part_locs = torch.from_numpy(all_part_locs).to(device)

    # Only evaluate the part attributes
    all_loc_acc, all_attri_idx, all_num_samples = [], [], []
    for attri_idx in tqdm(range(n_attributes)):
        attribute_name = attributes_names[attri_idx]  # The name of current attribute
        if attribute_name not in part_attributes_names:  # Only evaluate the part attributes, eliminate the attributes for the whole body
            continue
        part_indexes = torch.LongTensor(part_name_to_part_indexes[attribute_name]).to(device)

        attri_labels = all_attributes[:, attri_idx]
        select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(
            dim=1)  # Select all the test images containing this attribute

        n_select_samples = select_img_indexes.shape[0]
        select_activaton_maps = all_activation_maps[attri_idx]  # (n_select_samples, fea_h, fea_w)

        # Get the activation maps
        select_activaton_maps = select_activaton_maps[:, None]  # (n_select_samples, 1, fea_h, fea_w)
        upsampled_activation_maps = torch.nn.functional.interpolate(select_activaton_maps, size=(img_size, img_size),
                                                                    mode='bicubic')
        upsampled_activation_maps = upsampled_activation_maps.squeeze(dim=1)  # (n_select_samples, img_h, img_w)

        # Get the prediction bboxes
        max_indice = upsampled_activation_maps.flatten(start_dim=1).argmax(dim=-1)
        mi_h, mi_w = torch.div(max_indice, img_size, rounding_mode='trunc'), max_indice % img_size
        bhz = bbox_half_size
        pred_y1, pred_y2, pred_x1, pred_x2 = torch.where(mi_h - bhz >= 0, mi_h - bhz, 0), \
                                             torch.where(mi_h + bhz <= img_size, mi_h + bhz, img_size), \
                                             torch.where(mi_w - bhz >= 0, mi_w - bhz, 0), \
                                             torch.where(mi_w + bhz <= img_size, mi_w + bhz, img_size)
        pred_bboxes = torch.stack([pred_y1, pred_y2, pred_x1, pred_x2], dim=1)  # (n_select_samples, 4)

        # Get the ground-truth part locations
        part_locs = all_part_locs[select_img_indexes]  # (n_select_samples, 15, 2)
        part_indexes = part_indexes[None, :, None].repeat(n_select_samples, 1, 2)
        part_locs = torch.gather(part_locs, 1,
                                 part_indexes)  # (n_select_samples, Np, 2), Np is the number of part location annotations of current attribute in the image
        part_exist = part_locs.sum(dim=-1) > 0  # (n_select_samples, Np)
        sample_exist = part_exist.sum(dim=-1) > 0  # (n_select_samples)
        cal_img_indexes = torch.nonzero(sample_exist == 1).squeeze(
            dim=1)  # The images without part location annotations are eliminated

        # 确保索引和被索引的张量在同一设备上
        cal_img_indexes = cal_img_indexes.to(part_locs.device)

        # n_cal_samples = cal_img_indexes.shape[0]
        cal_pred_bboxes, cal_part_locs, cal_part_exist = pred_bboxes[cal_img_indexes], part_locs[cal_img_indexes], \
                                                         part_exist[
                                                             cal_img_indexes]  # (n_cal_samples, 4), (n_cal_samples, Np, 2), (n_cal_samples, Np)

        cal_pred_bboxes = cal_pred_bboxes[:, None]  # (n_cal_samples, 1, 4)
        cal_cond1, cal_cond2, cal_cond3, cal_cond4 = cal_part_locs[:, :, 0] - cal_pred_bboxes[:, :, 0] >= 0, \
                                                     cal_part_locs[:, :, 0] - cal_pred_bboxes[:, :, 1] <= 0, \
                                                     cal_part_locs[:, :, 1] - cal_pred_bboxes[:, :, 2] >= 0, \
                                                     cal_part_locs[:, :, 1] - cal_pred_bboxes[:, :,
                                                                              3] <= 0  # Each one: (n_cal_samples, Np)
        cal_part_inside = torch.stack([cal_cond1, cal_cond2, cal_cond3, cal_cond4], dim=2).sum(
            dim=-1) == 4  # (n_cal_samples, Np), estimate whether the ground-truth part location is inside the prediction bbox
        cal_part_inside = cal_part_inside.sum(dim=-1) > 0  # (n_cal_samples,)

        loc_acc = cal_part_inside.sum(dim=0) / cal_part_inside.shape[
            0]  # Calculate the ratio of images that the ground-truth part location is inside the prediction bbox

        all_loc_acc.append(loc_acc.item())
        all_attri_idx.append(attri_idx)
        all_num_samples.append(n_select_samples)
    all_loc_acc = np.array(all_loc_acc)
    all_attri_idx = np.array(all_attri_idx)
    all_num_samples = np.array(all_num_samples)
    mean_loc_acc = all_loc_acc.mean()

    return mean_loc_acc * 100, (all_loc_acc, all_attri_idx, all_num_samples)


def generate_attribute_activation_maps(all_concept_acts, all_attributes, all_img_ids, c2a_weight=None, model=None,
                                       device='cuda'):
    """
    为每个属性生成激活图，适用于概念数等于属性数的情况

    参数:
        all_concept_acts: 所有概念的激活图 [n_samples, n_concepts, fea_h, fea_w]
        all_attributes: 所有样本的属性标签 [n_samples, n_attributes]
        all_img_ids: 所有样本的图像ID
        c2a_weight: 概念到属性的权重矩阵，如果为None则使用一对一映射
        model: 模型对象，用于获取权重矩阵(如果c2a_weight为None)
        device: 计算设备 ('cuda' 或 'cpu')

    返回:
        all_activation_maps: 每个属性的激活图列表
        其他评估所需的指标
    """
    # 确保设备名称有效
    if device == 'gpu':
        device = 'cuda'

    # 确定所有张量的设备
    concept_acts_device = all_concept_acts.device
    attributes_device = all_attributes.device

    # 打印设备信息以帮助调试
    print(f"概念激活图设备: {concept_acts_device}, 属性标签设备: {attributes_device}")

    # 确保所有张量在同一设备上
    all_concept_acts = all_concept_acts.to(concept_acts_device)
    all_attributes = all_attributes.to(concept_acts_device)

    n_samples, n_concepts, fea_h, fea_w = all_concept_acts.shape
    n_attributes = all_attributes.shape[1]

    # 确定方法：一对一映射或权重选择
    if n_concepts == n_attributes and c2a_weight is None:
        # 方法1：一对一映射 - 每个概念直接对应一个属性
        print("使用一对一映射从概念到属性")
        use_one_to_one = True
        # 在这种情况下，我们不需要corre_proto_indexes
        corre_proto_indexes = torch.arange(n_attributes).unsqueeze(1)  # [n_attributes, 1]
    else:
        # 方法2：基于权重选择相关概念
        use_one_to_one = False

        # 如果没有提供权重矩阵，尝试从模型获取或创建一个
        if c2a_weight is None:
            if model is not None and hasattr(model, 'c2y_model'):
                print("从模型的c2y_model获取概念-属性权重")
                c2a_weight = model.c2y_model[0].weight.detach()  # [n_attributes, n_concepts]
            else:
                print("创建一个简单的一对一映射权重矩阵")
                # 创建单位矩阵作为权重（如果属性数等于概念数）
                c2a_weight = torch.eye(n_attributes, n_concepts)

        # 确定每个属性对应的概念数量
        if n_concepts == n_attributes:
            # 当概念数等于属性数时，每个属性选择较少的相关概念
            corre_proto_num = 3  # 每个属性选择3个最相关的概念
        elif n_concepts > n_attributes:
            # 当概念数多于属性数时，每个属性可以选择更多的相关概念
            corre_proto_num = min(10, n_concepts // n_attributes * 3)
        else:
            # 当概念数少于属性数时，每个属性可能只对应1个概念
            corre_proto_num = 1

        print(f"每个属性选择{corre_proto_num}个最相关的概念")

        # 为每个属性找出最相关的概念
        a_highest_indexes = torch.argsort(c2a_weight, dim=1, descending=True)
        corre_proto_indexes = a_highest_indexes[:, :corre_proto_num]  # [n_attributes, corre_proto_num]

    # 为每个属性生成激活图
    all_activation_maps = []  # 每个属性的激活图列表
    for attri_idx in tqdm(range(n_attributes), desc="生成属性激活图"):
        # 选择包含该属性的所有测试图像
        attri_labels = all_attributes[:, attri_idx]
        select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(dim=1)

        # 确保索引与all_concept_acts在同一设备上
        select_img_indexes = select_img_indexes.to(concept_acts_device)

        if len(select_img_indexes) == 0:
            # 如果没有样本包含该属性，跳过
            all_activation_maps.append(torch.zeros(0, fea_h, fea_w, device=concept_acts_device))
            continue

        if use_one_to_one:
            # 方法1：直接使用对应概念的激活图
            try:
                activation_maps = all_concept_acts[select_img_indexes, attri_idx]  # [n_select_samples, fea_h, fea_w]
            except RuntimeError as e:
                print(f"错误：{e}")
                print(
                    f"设备信息: all_concept_acts: {all_concept_acts.device}, select_img_indexes: {select_img_indexes.device}")
                # 尝试将索引移到CPU上
                select_img_indexes_cpu = select_img_indexes.cpu()
                all_concept_acts_cpu = all_concept_acts.cpu()
                activation_maps = all_concept_acts_cpu[select_img_indexes_cpu, attri_idx].to(concept_acts_device)
        else:
            # 方法2：选择多个相关概念的激活图并平均
            cur_corre_proto_indexes = corre_proto_indexes[attri_idx]

            # 确保索引在正确的设备上
            cur_corre_proto_indexes = cur_corre_proto_indexes.to(concept_acts_device)

            try:
                select_proto_acts = all_concept_acts[select_img_indexes][:,
                                    cur_corre_proto_indexes]  # [n_select_samples, corre_proto_num, fea_h, fea_w]
                activation_maps = select_proto_acts.mean(dim=1)  # [n_select_samples, fea_h, fea_w]
            except RuntimeError as e:
                print(f"错误：{e}")
                print(
                    f"设备信息: all_concept_acts: {all_concept_acts.device}, select_img_indexes: {select_img_indexes.device}, cur_corre_proto_indexes: {cur_corre_proto_indexes.device}")
                # 尝试将所有张量移到CPU上进行索引
                select_img_indexes_cpu = select_img_indexes.cpu()
                cur_corre_proto_indexes_cpu = cur_corre_proto_indexes.cpu()
                all_concept_acts_cpu = all_concept_acts.cpu()

                select_proto_acts = all_concept_acts_cpu[select_img_indexes_cpu][:, cur_corre_proto_indexes_cpu]
                activation_maps = select_proto_acts.mean(dim=1).to(concept_acts_device)

        all_activation_maps.append(activation_maps)

    return all_activation_maps


if __name__ == '__main__':
    args = get_args()
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    with open(f"configs/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    experiment_config["model_pretrain_path"] = "./checkpoints/labeled-ratio-80/test.pt"

    dataset_config = experiment_config['dataset_config']
    if args.dataset == "CUB-200-2011":
        data_module = cub_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")

    results = defaultdict(dict)
    for current_config in experiment_config['runs']:
        run_name = current_config['architecture']
        trial_config = copy.deepcopy(experiment_config)
        trial_config.update(current_config)

        for run_config in generate_hyper_param_configs(trial_config):
            run_config = copy.deepcopy(run_config)
            run_config['result_dir'] = save_dir
            evaluate_expressions(run_config, soft=True)

            model = construct_model(112, 200, run_config)
            if run_config.get("model_pretrain_path"):
                if os.path.exists(run_config.get("model_pretrain_path")):
                    logging.info("Load pretrained model")
                    model.load_state_dict(torch.load(run_config.get("model_pretrain_path")), strict=False)
            model.eval()
            grad_cam = CBMGradCAM(model)

            transform = transforms.Compose([
                transforms.CenterCrop(299),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])

            root_dir = './data/CUB_200_2011'
            base_dir = os.path.join(root_dir, 'class_attr_data_10')
            train_data_path = os.path.join(base_dir, 'train.pkl')
            val_data_path = os.path.join(base_dir, 'val.pkl')
            test_data_path = os.path.join(base_dir, 'test.pkl')

            dataset = CUBDataset_for_heatmap(
                pkl_file_paths=[train_data_path],
                image_dir='images',
                transform=transform,
                root_dir=root_dir,
            )

            loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=False, num_workers=64)

            all_concept_acts = []
            all_targets = []
            all_img_ids = []
            concept_set = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
            for b_idx, batch in enumerate(tqdm(loader, desc="heatmap generating")):
                data, targets, img_ids = batch
                heat = model.plot_heatmap(data)
                all_concept_acts.append(heat)
                all_targets.append(targets)
                all_img_ids.append(img_ids)
                break
            all_concept_acts = torch.cat(all_concept_acts, dim=0)  # [n_samples, n_concepts, fea_h, fea_w]
            all_targets = torch.cat(all_targets, dim=0)
            all_img_ids = torch.cat(all_img_ids, dim=0)

            print("finish generating")

            all_attributes = []
            for img_id in all_img_ids:
                attributes = id_to_attributes[img_id.item()]
                all_attributes.append(attributes)
            all_attributes = np.stack(all_attributes, axis=0)
            all_attributes = torch.from_numpy(all_attributes).cuda()
            n_attributes = all_attributes.shape[-1]

            print("finish attributes")

            all_activation_maps = generate_attribute_activation_maps(
                all_concept_acts, all_attributes, all_img_ids, model=model)

            print("finish generating maps")

            mean_loc_acc, (all_loc_acc, all_attri_idx, all_num_samples) = evaluate_concept_trustworthiness(
                all_activation_maps,
                all_img_ids,
                bbox_half_size=45)
            attributes_names = np.array(attributes_names)
            select_attribute_names = attributes_names[all_attri_idx]

            np.set_printoptions(precision=2)
            print('Mean Loc Accuracy : %.2f' % (mean_loc_acc))

            exit(0)

            activation_generator = CBMActivationMapGenerator(model)

            # 生成所有样本的概念激活图
            all_proto_acts, all_targets, all_img_ids = activation_generator.batch_generate_gradcam(loader, device)

            # 清理钩子
            activation_generator.remove_hooks()

            # Get the ground-truth attributes of the all test images
            all_attributes = []
            for img_id in all_img_ids:
                attributes = id_to_attributes[img_id.item()]
                all_attributes.append(attributes)
            all_attributes = np.stack(all_attributes, axis=0)
            all_attributes = torch.from_numpy(all_attributes).cuda()
            n_attributes = all_attributes.shape[-1]

            # # Get the averaged activation map of each attribute
            # corre_proto_num = 10
            # a_weight = ppnet.attributes_predictor.weight.detach()  # (112, 2000)
            # a_highest_indexes = torch.argsort(a_weight, dim=1,
            #                                   descending=True)  # (112, 2000), Get the indexes of the sorted weights
            # corre_proto_indexes = a_highest_indexes[:, :corre_proto_num]  # (112, corre_proto_num)
            #
            # all_activation_maps = []  # all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute, the length is 112
            # for attri_idx in tqdm(range(n_attributes)):
            #     attri_labels = all_attributes[:, attri_idx]
            #     select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(
            #         dim=1)  # Select all the test images containing this attribute
            #     cur_corre_proto_indexes = corre_proto_indexes[attri_idx]
            #
            #     select_proto_acts = all_proto_acts[select_img_indexes][:,
            #                         cur_corre_proto_indexes]  # (n_select_samples, corre_proto_num, fea_h, fea_w)
            #     activation_maps = select_proto_acts.mean(dim=1)  # (n_select_samples, fea_h, fea_w)
            #     all_activation_maps.append(activation_maps)

            all_activation_maps = generate_attribute_activation_maps(
                all_proto_acts, all_attributes, all_img_ids, model=model)

            mean_loc_acc, (all_loc_acc, all_attri_idx, all_num_samples) = evaluate_concept_trustworthiness(
                all_activation_maps,
                all_img_ids,
                bbox_half_size=45)
            attributes_names = np.array(attributes_names)
            select_attribute_names = attributes_names[all_attri_idx]

            np.set_printoptions(precision=2)
            print('Mean Loc Accuracy : %.2f' % (mean_loc_acc))

            concept_set = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
            for b_idx, batch in enumerate(loader):
                x, x_show, y, c, img_name = batch
                model.plot_heatmap(x, x_show, c, y, img_name, f"{save_dir}/heatmap", concept_set)
                break

    print(f"========================finish========================")
