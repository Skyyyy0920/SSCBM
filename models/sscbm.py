import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision.models import resnet50, resnet34
from models.cbm import CBM_SSL
import train.utils as utils
from utils import visualize_and_save_heatmaps
from cem.metrics.accs import compute_accuracy


class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h, w = x.shape

        x = self.proj_in(x)  # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1,
                                                                         2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)  # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
        out = self.proj_out(out)  # [batch_size, c, h, w]

        return out, att_weights


class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h, w = x.shape

        x = self.proj_in(x)  # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1,
                                                                         2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)  # [batch_size, h*w, emb_dim]

        print(out.shape)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
        out = self.proj_out(out)  # [batch_size, c, h, w]

        return out, att_weights


class FeatureExtractor(nn.Module):
    """修改ResNet提取器，保留空间特征信息"""

    def __init__(self, output_dim=None):
        super().__init__()
        # 加载不带顶层的ResNet
        resnet = resnet34(pretrained=True)
        # 移除最后的全局平均池化和全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_features = 512

    def forward(self, x):
        # 返回特征图 [B, 2048, H, W] 通常是 [B, 2048, 7, 7]
        return self.features(x)


class SpatialCrossAttention(nn.Module):
    """空间感知的交叉注意力模块"""

    def __init__(self, embed_dim, feature_channels, use_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads

        # 投影层
        self.feature_proj = nn.Conv2d(feature_channels, embed_dim, kernel_size=1)
        self.concept_query_proj = nn.Linear(embed_dim, embed_dim)

        # 多头注意力
        self.mh_concept = nn.Linear(embed_dim, self.num_heads * self.head_dim)
        self.mh_feature = nn.Linear(embed_dim, 2 * self.num_heads * self.head_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

        # 概念得分预测
        self.score_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features, c_embedding):
        """
        Args:
            features: [B, C, H, W] 特征图
            c_embedding: [B, N, D] 概念嵌入
        Returns:
            concept_scores: [B, N] 概念得分
            concept_maps: [B, N, H, W] 概念激活热图
        """
        B, C, H, W = features.shape
        N = c_embedding.size(1)  # 概念数量

        # 投影特征图
        feature_proj = self.feature_proj(features)  # [B, D, H, W]
        feature_proj = feature_proj.flatten(2).permute(0, 2, 1)  # [B, H*W, D]

        # 投影概念查询
        concept_query = self.concept_query_proj(c_embedding)  # [B, N, D]

        # 多头注意力
        mh_query = self.mh_concept(concept_query).view(
            B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]

        mh_key_value = self.mh_feature(feature_proj)  # [B, H*W, 2*h*d]
        mh_key, mh_value = mh_key_value.chunk(2, dim=-1)
        mh_key = mh_key.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, H*W, d]
        mh_value = mh_value.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, H*W, d]

        # 计算注意力权重
        attn_logits = torch.matmul(mh_query, mh_key.transpose(-2, -1))  # [B, h, N, H*W]
        attn_logits = attn_logits / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, h, N, H*W]
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

        # 这个权重矩阵就是我们需要的热图基础
        # 把它重新整形为空间特征图
        concept_maps = attn_weights.mean(dim=1)  # [B, N, H*W]
        concept_maps = concept_maps.view(B, N, H, W)  # [B, N, H, W]

        # 计算加权特征
        attended = torch.matmul(attn_weights, mh_value)  # [B, h, N, d]
        attended = attended.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # [B, N, D]

        if self.use_residual:
            attended = attended + c_embedding

        attended = self.layer_norm(attended)

        # 计算概念得分
        concept_scores = self.score_proj(attended).squeeze(-1)  # [B, N]

        return concept_scores, concept_maps


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, image_feature_dim, use_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads

        self.image_proj = nn.Linear(image_feature_dim, embed_dim)
        self.concept_query_proj = nn.Linear(embed_dim, embed_dim)

        self.mh_concept = nn.Linear(embed_dim, self.num_heads * self.head_dim)
        self.mh_image = nn.Linear(embed_dim, 2 * self.num_heads * self.head_dim)

        self.res_gate = nn.Linear(embed_dim, embed_dim) if use_residual else None

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.score_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)

        nn.init.xavier_uniform_(self.concept_query_proj.weight)
        nn.init.zeros_(self.concept_query_proj.bias)

        nn.init.normal_(self.mh_concept.weight, std=0.02)
        nn.init.normal_(self.mh_image.weight, std=0.02)

        if self.res_gate:
            nn.init.constant_(self.res_gate.weight, 0.)
            nn.init.constant_(self.res_gate.bias, 1.)

    def forward(self, image_feature, c_embedding):
        B, _ = image_feature.shape  # [B, image_feature_dim]
        N = c_embedding.size(1)  # [B, N, D]

        image_feature = self.image_proj(image_feature)  # [B, D]
        image_feature = image_feature.unsqueeze(1)  # [B, 1, D]

        concept_query = self.concept_query_proj(c_embedding)  # [B, N, D]

        mh_query = self.mh_concept(concept_query).view(
            B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]

        mh_key_value = self.mh_image(image_feature)  # [B, 1, 2*h*d]
        mh_key, mh_value = mh_key_value.chunk(2, dim=-1)
        mh_key = mh_key.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, 1, d]
        mh_value = mh_value.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, 1, d]

        attn_logits = torch.matmul(mh_query, mh_key.transpose(-2, -1))  # [B, h, N, 1]
        attn_logits = attn_logits / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

        attended = torch.matmul(attn_weights, mh_value)  # [B, h, N, d]
        attended = attended.permute(0, 2, 1, 3).contiguous().view(B, N, -1)  # [B, N, D]

        if self.use_residual:
            gate = torch.sigmoid(self.res_gate(attended))
            attended = gate * attended + (1 - gate) * c_embedding
        else:
            attended = attended + c_embedding

        attended = self.layer_norm(attended)

        concept_scores = self.score_proj(attended).squeeze(-1)  # [B, N]

        return concept_scores


class SSCBM(CBM_SSL):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            shared_prob_gen=True,
            concept_loss_weight=1,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,

            c2y_model=None,
            c2y_layers=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            output_latent=False,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,
            tau=1,

            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,
            use_concept_groups=False,

            top_k_accuracy=None,
    ):
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        # self.pre_concept_model = c_extractor_arch(output_dim=None)

        self.pre_concept_model = FeatureExtractor(output_dim=None)
        self.resnet_out_features = self.pre_concept_model.out_features

        # 添加池化层，用于生成向量表示
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)

        # 添加空间交叉注意力模块
        self.spatial_cross_attn = SpatialCrossAttention(
            embed_dim=emb_size,
            feature_channels=self.resnet_out_features
        )

        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        # self.resnet_out_features = list(self.pre_concept_model.modules())[-1].out_features
        self.resnet_out_features = 1000

        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[torch.nn.Linear(self.resnet_out_features, 2 * emb_size)])
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(self.resnet_out_features, 2 * emb_size),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and len(self.concept_prob_generators) == 0:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))

        if c2y_model is None:
            units = [n_concepts * emb_size] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        # self.cross_attn = CrossAttention(emb_size, self.resnet_out_features)

        self.sigmoid = torch.nn.Sigmoid()

        self.loss_concept_labeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_concept_unlabeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights)
        )
        self.concept_loss_weight_labeled = concept_loss_weight_labeled
        self.concept_loss_weight_unlabeled = concept_loss_weight_unlabeled
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.tau = tau
        self.use_concept_groups = use_concept_groups

        # self.fc = nn.Linear(512, self.emb_size)
        # self.pooling = nn.AdaptiveAvgPool2d(1)

    def _after_interventions(
            self,
            prob,
            pos_embeddings,
            neg_embeddings,
            intervention_idxs=None,
            c_true=None,
            train=False,
            competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
                (c_true is not None) and
                (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return prob * (1 - intervention_idxs) + intervention_idxs * c_true, intervention_idxs

    def _forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            train=False,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None
    ):
        if latent is None:
            # pre_c = self.pre_concept_model(x)  # [batch_size, 299, 299] -> [batch_size, resnet_out_features]
            # contexts = []
            # c_sem = []

            features = self.pre_concept_model(x)  # [batch_size, 2048, H, W]

            # 保存原始特征图尺寸用于插值
            feat_h, feat_w = features.shape[2], features.shape[3]

            # 应用全局平均池化得到向量表示
            pooled_features = self.pooling(features).squeeze(-1).squeeze(-1)
            pooled_features = self.fc(pooled_features)

            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                # context = context_gen(pre_c)  # [batch_size, resnet_out_features] -> [batch_size, 2 * emb_size]
                context = context_gen(pooled_features)

                prob = prob_gen(context)  # [batch_size, 2 * emb_size] -> [batch_size, 1]
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sigmoid(prob))
            c_sem = torch.cat(c_sem, dim=-1)  # [batch_size, 1, n_concepts] -> [batch_size, n_concepts]
            contexts = torch.cat(contexts, dim=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        if (intervention_idxs is None) and (c is not None) and (self.intervention_policy is not None):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=None,
            )
        else:
            c_int = c

        if not train:
            intervention_idxs = self._standardize_indices(intervention_idxs=intervention_idxs, batch_size=x.shape[0])

        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )

        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )  # [batch_size, n_concepts, D]
        c_pred = c_embedding.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)

        image_feature = self.pre_concept_model(x)  # [batch_size, resnet_out_features]
        # c_pred_unlabeled = self.cross_attn(image_feature, c_embedding)
        c_pred_unlabeled, concept_maps = self.spatial_cross_attn(features, c_embedding)

        tail_results = []
        if output_interventions:
            print(f"output_intervention")
            if intervention_idxs is not None and isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(intervention_idxs).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            print(f"output_latent")
            tail_results.append(latent)
        if output_embeddings:
            print(f"output_embedding")
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])

        return tuple([c_sem, c_pred, c_pred_unlabeled, y] + tail_results)

    def _run_step(
            self,
            batch,
            batch_idx,
            train=False,
            intervention_idxs=None,
    ):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)

        nbr_w_ = nbr_w.unsqueeze(-1).repeat(1, 1, nbr_c.size(2))
        c_pseudo = nbr_c * nbr_w_
        c_pseudo = torch.sum(c_pseudo, dim=1) / nbr_w.size(1)

        outputs = self._forward(
            x,
            c=c,
            y=y,
            l=l,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
        )
        c_sem, c_pred_labeled, c_pred_unlabeled, y_pred = outputs[0], outputs[1], outputs[2], outputs[3]

        task_loss = self.loss_task(y_pred, y)
        task_loss_scalar = task_loss.detach()

        concept_loss_labeled = self.loss_concept_labeled(c_sem[l], c[l])
        concept_loss_scalar_labeled = concept_loss_labeled.detach()

        c_pred_unlabeled = c_pred_unlabeled.float()
        c_pseudo = c_pseudo.float()
        concept_loss_unlabeled = self.loss_concept_unlabeled(c_pred_unlabeled[~l], c_pseudo[~l])
        concept_loss_scalar_unlabeled = concept_loss_unlabeled.detach()

        loss = (task_loss
                + self.concept_loss_weight_labeled * concept_loss_labeled
                + self.concept_loss_weight_unlabeled * concept_loss_unlabeled)

        # compute accuracy
        (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_accuracy(c_sem, y_pred, c, y)
        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_acc,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "c_loss_labeled": concept_loss_scalar_labeled,
            "c_loss_unlabeled": concept_loss_scalar_unlabeled,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_acc + y_acc) / 2,
        }
        return loss, result

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"================================Epoch {self.current_epoch}===============================")
        loss, result = self._run_step(batch, batch_idx, train=True)
        for name, val in result.items():
            if name in ['c_f1', 'y_auc', 'avg_c_y_acc', 'y_f1']:
                continue
            self.log(name, val, prog_bar=True)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_acc'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_acc'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss_labeled": result['c_loss_labeled'],
                "concept_loss_unlabeled": result['c_loss_unlabeled'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def predict_step(
            self,
            batch,
            batch_idx,
            intervention_idxs=None,
            dataloader_idx=0,
    ):
        x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = self._unpack_batch(batch)
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            l=l,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_interventions=True
        )

    def plot_heatmap(
            self,
            x,
            x_show=None,
            c=None,
            y=None,
            img_name=None,
            output_dir='heatmap',
            concept_set=None,
    ):
        pre_c = self.image_encoder(x)
        contexts = []
        c_sem = []

        for i, context_gen in enumerate(self.concept_context_generators):
            if self.shared_prob_gen:
                prob_gen = self.concept_prob_generators[0]
            else:
                prob_gen = self.concept_prob_generators[i]
            context = context_gen(pre_c)
            prob = prob_gen(context)
            contexts.append(torch.unsqueeze(context, dim=1))
            c_sem.append(self.sigmoid(prob))
        c_sem = torch.cat(c_sem, dim=-1)
        contexts = torch.cat(contexts, dim=1)

        probs = c_sem
        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        image_feature = self.unlabeled_image_encoder(x)

        heatmap = []
        for i in range(len(image_feature)):
            heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)

        for i in range(len(x)):
            # save_dir = f"./{output_dir}/{img_name[i]}"
            save_dir = f"/root/autodl-tmp/heatmap/{img_name[i]}"

            visualize_and_save_heatmaps(x_show[i], c[i], c_sem[i], heatmap[i], save_dir, concept_set)
