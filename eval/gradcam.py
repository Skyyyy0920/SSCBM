import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import functional as TF


class CBMGradCAM:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.gradients = None
        self.activations = None

        if hasattr(model.pre_concept_model, 'layer4'):
            self.target_layer = model.pre_concept_model.layer4[-1].conv3
        else:  # If not resnet architecture
            modules = list(model.pre_concept_model.modules())
            for i in reversed(range(len(modules))):
                if isinstance(modules[i], torch.nn.Conv2d):
                    self.target_layer = modules[i]
                    break

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)

        self.hooks = [forward_handle, backward_handle]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, input_image, concept_idx):
        """
        为指定概念生成类激活映射

        参数:
            input_image: 输入图像张量 [1, C, H, W]
            concept_idx: 要可视化的概念索引

        返回:
            cam: 类激活映射的numpy数组
        """
        self.model.eval()
        self.model.zero_grad()

        # 前向传播
        outputs = self.model._forward(input_image)
        concept_probs = outputs[0]  # 假设第一个输出是概念概率

        # 选择目标概念
        concept_score = concept_probs[0, concept_idx]

        # 反向传播
        self.model.zero_grad()
        concept_score.backward(retain_graph=True)

        # 生成加权激活图
        gradients = self.gradients.data.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.data.cpu().numpy()[0]  # [C, H, W]

        # 计算梯度的全局平均值作为权重
        weights = np.mean(gradients, axis=(1, 2))  # [C]

        # 通过权重对激活图进行加权求和
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 应用ReLU，只关注正向影响
        cam = np.maximum(cam, 0)

        # 归一化到0-1之间
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam

    def overlay_heatmap(self, image, cam, alpha=0.5):
        """
        将CAM热力图叠加到原始图像上

        参数:
            image: 原始图像张量
            cam: 类激活映射
            alpha: 热力图在叠加图像中的权重

        返回:
            overlay: 叠加后的图像numpy数组
        """
        # 将图像张量转换为numpy数组
        if isinstance(image, torch.Tensor):
            # 处理批次维度和通道优先格式
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.cpu().permute(1, 2, 0).numpy()

            # 如果需要，将图像归一化到0-255范围
            if image.max() <= 1.0:
                image = image * 255

        # 将CAM调整为图像大小
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # 应用颜色映射到CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 将热力图叠加到原始图像上
        overlay = heatmap * alpha + image * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay


def visualize_concept_maps(model, image, concept_indices=None, concept_names=None, figsize=(12, 10)):
    """
    可视化多个概念的Grad-CAM热力图

    参数:
        model: CBM模型
        image: 输入图像张量 [1, C, H, W]
        concept_indices: 要可视化的概念索引列表 (默认: 所有概念)
        concept_names: 可选的概念名称列表
        figsize: 图像大小
    """
    model.eval()
    grad_cam = CBMGradCAM(model)

    # 确定要可视化的概念
    if concept_indices is None:
        concept_indices = list(range(model.n_concepts))

    # 获取概念概率
    with torch.no_grad():
        outputs = model._forward(image)
        concept_probs = outputs[0].cpu().numpy()[0]

    # 确定网格布局
    n_concepts = len(concept_indices)
    n_cols = min(5, n_concepts + 1)  # +1 用于原始图像
    n_rows = (n_concepts + n_cols) // n_cols

    plt.figure(figsize=figsize)

    # 显示原始图像
    if isinstance(image, torch.Tensor):
        orig_img = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # 如果需要去归一化
        # orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)
    else:
        orig_img = image

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(orig_img)
    plt.title('原始图像')
    plt.axis('off')

    # 为每个概念生成并显示CAM
    for i, concept_idx in enumerate(concept_indices):
        # 生成CAM
        cam = grad_cam.generate_cam(image, concept_idx)

        # 创建叠加图
        overlay = grad_cam.overlay_heatmap(image, cam, alpha=0.5)

        # 显示
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(overlay)

        # 获取概念名称和概率
        concept_name = concept_names[concept_idx] if concept_names else f"概念 {concept_idx}"
        prob = concept_probs[concept_idx]

        plt.title(f"{concept_name}\n(概率={prob:.2f})")
        plt.axis('off')

    grad_cam.remove_hooks()
    plt.tight_layout()
    plt.show()


class CBMActivationMapGenerator:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.feature_maps = None

        # 对于ResNet50，最后一个卷积层通常在layer4中
        # 需要获取池化前的特征图
        if hasattr(model.pre_concept_model, 'layer4'):
            # 识别ResNet架构类型（BasicBlock或Bottleneck）
            last_block = model.pre_concept_model.layer4[-1]
            if hasattr(last_block, 'conv3'):
                # Bottleneck结构(ResNet50/101/152)
                self.target_layer = last_block.conv3
            elif hasattr(last_block, 'conv2'):
                # BasicBlock结构(ResNet18/34)
                self.target_layer = last_block.conv2
        else:
            # 如果不是ResNet架构，寻找最后一个卷积层
            modules = list(model.pre_concept_model.modules())
            for i in reversed(range(len(modules))):
                if isinstance(modules[i], torch.nn.Conv2d):
                    self.target_layer = modules[i]
                    break

        # 注册钩子来获取特征图
        self._register_forward_hook()

    def _register_forward_hook(self):
        def hook_fn(module, input, output):
            self.feature_maps = output

        handle = self.target_layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def remove_hooks(self):
        """移除钩子，防止内存泄漏"""
        for hook in self.hooks:
            hook.remove()

    def generate_concept_activation_maps(self, dataloader, device='cuda'):
        """
        为数据集中的所有样本生成所有概念的激活图

        参数:
            dataloader: 包含数据的DataLoader
            device: 计算设备

        返回:
            all_concept_acts: 概念激活图 [n_samples, n_concepts, fea_h, fea_w]
            all_targets: 目标标签
            all_img_ids: 图像ID (如果dataloader包含)
        """
        self.model.eval()
        all_concept_acts = []
        all_targets = []
        all_img_ids = []

        # 临时存储概念权重
        n_concepts = self.model.n_concepts
        concept_weights = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="generate concept heatmap")):
                # 处理不同的dataloader格式
                if len(batch) == 3:  # (data, targets, img_ids)
                    data, targets, img_ids = batch
                elif len(batch) == 2:  # (data, targets)
                    data, targets = batch
                    img_ids = torch.arange(data.shape[0]) + batch_idx * data.shape[0]

                data = data.to(device)
                targets = targets.to(device)

                # 前向传播，获取特征图
                _ = self.model._forward(data)
                feature_maps = self.feature_maps  # [batch_size, channels, h, w]

                # 获取每个概念的权重
                batch_concept_maps = []
                for c_idx in range(n_concepts):
                    # 如果是第一个批次，计算并存储每个概念的权重
                    if c_idx not in concept_weights:
                        # 计算该概念相对于特征图的梯度
                        self.model.zero_grad()
                        feature_maps.retain_grad()

                        # 对单一数据样本运行模型来获取梯度
                        single_data = data[0:1]
                        outputs = self.model._forward(single_data)
                        concept_prob = outputs[0][0, c_idx]
                        concept_prob.backward(retain_graph=(c_idx < n_concepts - 1))

                        # 获取梯度并计算通道权重
                        gradients = feature_maps.grad[0].mean(dim=[1, 2])  # [channels]
                        concept_weights[c_idx] = gradients

                    # 使用预计算的权重生成激活图
                    weights = concept_weights[c_idx].unsqueeze(-1).unsqueeze(-1)  # [channels, 1, 1]
                    concept_map = (weights * feature_maps).sum(dim=1)  # [batch_size, h, w]

                    # 应用ReLU以只关注正向影响
                    concept_map = F.relu(concept_map)

                    batch_concept_maps.append(concept_map.unsqueeze(1))  # [batch_size, 1, h, w]

                # 连接所有概念的激活图
                batch_concept_maps = torch.cat(batch_concept_maps, dim=1)  # [batch_size, n_concepts, h, w]
                all_concept_acts.append(batch_concept_maps)
                all_targets.append(targets)
                all_img_ids.append(img_ids)

        # 连接所有批次
        all_concept_acts = torch.cat(all_concept_acts, dim=0)  # [n_samples, n_concepts, fea_h, fea_w]
        all_targets = torch.cat(all_targets, dim=0)
        all_img_ids = torch.cat(all_img_ids, dim=0)

        return all_concept_acts, all_targets, all_img_ids

    def batch_generate_gradcam(self, dataloader, device='cuda'):
        """
        使用Grad-CAM为数据集中的所有样本生成所有概念的激活图

        参数:
            dataloader: 包含数据的DataLoader
            device: 计算设备

        返回:
            all_concept_acts: 概念激活图 [n_samples, n_concepts, fea_h, fea_w]
            all_targets: 目标标签
            all_img_ids: 图像ID (如果dataloader包含)
        """
        self.model.eval()
        all_concept_acts = []
        all_targets = []
        all_img_ids = []

        n_concepts = self.model.n_concepts

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="生成Grad-CAM")):
            data, targets, img_ids = batch

            data = data.to(device)
            targets = targets.to(device)

            # 获取批次大小
            batch_size = data.shape[0]

            # 为批次中的每个样本生成所有概念的激活图
            batch_concept_maps = []
            for concept_idx in range(n_concepts):
                sample_concept_maps = []

                for sample_idx in range(batch_size):
                    sample_data = data[sample_idx:sample_idx + 1]

                    # 前向传播
                    self.model.zero_grad()
                    _ = self.model._forward(sample_data)
                    feature_maps = self.feature_maps  # [1, channels, h, w]

                    # 重新计算概念概率并反向传播
                    self.model.zero_grad()
                    feature_maps.retain_grad()
                    outputs = self.model._forward(sample_data)
                    concept_prob = outputs[0][0, concept_idx]
                    concept_prob.backward(retain_graph=(sample_idx < batch_size - 1 or concept_idx < n_concepts - 1))

                    # 获取梯度
                    gradients = feature_maps.grad[0]  # [channels, h, w]

                    # 计算通道权重
                    weights = gradients.mean(dim=[1, 2])  # [channels]

                    # 生成激活图
                    weights = weights.unsqueeze(-1).unsqueeze(-1)  # [channels, 1, 1]
                    cam = (weights * feature_maps[0]).sum(dim=0)  # [h, w]

                    # 应用ReLU
                    cam = F.relu(cam)

                    # 归一化
                    if cam.max() > 0:
                        cam = cam / cam.max()

                    sample_concept_maps.append(cam.unsqueeze(0))  # [1, h, w]

                # 连接批次中所有样本的当前概念激活图
                concept_maps = torch.cat(sample_concept_maps, dim=0)  # [batch_size, h, w]
                batch_concept_maps.append(concept_maps.unsqueeze(1))  # [batch_size, 1, h, w]

            # 连接所有概念的激活图
            batch_concept_maps = torch.cat(batch_concept_maps, dim=1)  # [batch_size, n_concepts, h, w]
            all_concept_acts.append(batch_concept_maps)
            all_targets.append(targets)
            all_img_ids.append(img_ids)

        # 连接所有批次
        all_concept_acts = torch.cat(all_concept_acts, dim=0)  # [n_samples, n_concepts, fea_h, fea_w]
        all_targets = torch.cat(all_targets, dim=0)
        all_img_ids = torch.cat(all_img_ids, dim=0)

        return all_concept_acts, all_targets, all_img_ids


def generate_concept_heatmaps(model, test_loader, id_to_attributes=None, device='cuda'):
    """
    生成概念热力图并计算与属性的关联

    参数:
        model: SSCBM模型
        test_loader: 测试数据加载器
        id_to_attributes: 图像ID到属性的映射字典(如果有)
        device: 计算设备

    返回:
        热力图和相关计算结果
    """
    # 创建激活图生成器
    activation_generator = CBMActivationMapGenerator(model)

    # 生成所有样本的概念激活图
    print("生成概念激活图...")
    all_concept_acts, all_targets, all_img_ids = activation_generator.batch_generate_gradcam(
        test_loader, device=device)

    # 清理钩子
    activation_generator.remove_hooks()

    # 获取特征图尺寸
    fea_h, fea_w = all_concept_acts.shape[-2], all_concept_acts.shape[-1]
    print(f"特征图尺寸: {fea_h}×{fea_w}")

    # 如果提供了属性映射，计算属性相关内容
    if id_to_attributes is not None:
        # 获取所有测试图像的属性
        print("获取图像属性...")
        all_attributes = []
        for img_id in all_img_ids:
            attributes = id_to_attributes[img_id.item()]
            all_attributes.append(attributes)

        all_attributes = np.stack(all_attributes, axis=0)
        all_attributes = torch.from_numpy(all_attributes).to(device)
        n_attributes = all_attributes.shape[-1]

        # 从模型的概念到属性映射中获取权重
        print("计算概念-属性关联...")
        # 注意：此处假设SSCBM有一个属性预测器，如果没有，需要修改
        if hasattr(model, 'c2y_model'):
            a_weight = model.c2y_model[0].weight.detach()  # 假设第一层是Linear层
        else:
            # 如果没有属性预测器，创建一个简单的关联矩阵
            from sklearn.linear_model import LogisticRegression

            # 将激活图平均池化为概念特征
            concept_features = all_concept_acts.mean(dim=[2, 3]).cpu().numpy()
            attributes = all_attributes.cpu().numpy()

            # 为每个属性训练一个分类器
            a_weight = np.zeros((n_attributes, model.n_concepts))
            for attr_idx in range(n_attributes):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(concept_features, attributes[:, attr_idx])
                a_weight[attr_idx] = clf.coef_[0]

            a_weight = torch.from_numpy(a_weight).to(device)

        # 获取每个属性最相关的概念
        corre_proto_num = min(10, model.n_concepts)  # 每个属性最多关联10个概念
        a_highest_indexes = torch.argsort(a_weight, dim=1, descending=True)
        corre_proto_indexes = a_highest_indexes[:, :corre_proto_num]

        return all_concept_acts, all_targets, all_img_ids, all_attributes, corre_proto_indexes

    return all_concept_acts, all_targets, all_img_ids
