import torch
import logging
import sklearn.metrics
import pytorch_lightning as pl
from torchvision.models import resnet50, densenet121
import numpy as np

import train.utils as utils
from cem.metrics.accs import compute_accuracy


class CBM_SSL(pl.LightningModule):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            concept_loss_weight=1,
            concept_loss_weight_labeled=1,
            concept_loss_weight_unlabeled=5,
            task_loss_weight=1,

            extra_dims=0,
            bool=False,
            sigmoidal_prob=True,
            sigmoidal_extra_capacity=True,
            bottleneck_nonlinear=None,
            output_latent=False,

            x2c_model=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            c2y_model=None,
            c2y_layers=None,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            weight_loss=None,
            task_class_weights=None,

            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,
            use_concept_groups=False,

            top_k_accuracy=None,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        if x2c_model is not None:
            self.x2c_model = x2c_model
        else:
            self.x2c_model = c_extractor_arch(output_dim=(n_concepts + extra_dims))

        if c2y_model is not None:
            self.c2y_model = c2y_model
        else:
            units = [n_concepts + extra_dims] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        # Intervention-specific fields/handlers:
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(active_intervention_values)
        else:
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]) * (5.0 if not sigmoidal_prob else 1.0)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(inactive_intervention_values)
        else:
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]) * (-5.0 if not sigmoidal_prob else 0.0)

        self.sigmoid = torch.nn.Sigmoid()
        if sigmoidal_extra_capacity:
            bottleneck_nonlinear = "sigmoid"
        if bottleneck_nonlinear == "sigmoid":
            self.bottleneck_nonlin = torch.nn.Sigmoid()
        elif bottleneck_nonlinear == "leakyrelu":
            self.bottleneck_nonlin = torch.nn.LeakyReLU()
        elif bottleneck_nonlinear == "relu":
            self.bottleneck_nonlin = torch.nn.ReLU()
        elif (bottleneck_nonlinear is None) or (
                bottleneck_nonlinear == "identity"
        ):
            self.bottleneck_nonlin = lambda x: x
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{bottleneck_nonlinear}'"
            )

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.bool = bool
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.sigmoidal_prob = sigmoidal_prob
        self.sigmoidal_extra_capacity = sigmoidal_extra_capacity
        self.use_concept_groups = use_concept_groups

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            x, y, c = batch
            l, nbr_c, nbr_w = None, None, None
            competencies, prev_interventions = None, None
        elif len(batch) == 4:
            x, y, c, competencies = batch
            l, nbr_c, nbr_w, prev_interventions = None, None, None, None
        elif len(batch) == 5:
            x, y, c, competencies, prev_interventions = batch
            l, nbr_c, nbr_w = None, None, None
        elif len(batch) == 6:
            x, y, c, l, nbr_c, nbr_w = batch
            competencies, prev_interventions = None, None
        elif len(batch) == 7:
            x, y, c, l, nbr_c, nbr_w, competencies = batch
            prev_interventions = None
        else:
            x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions = batch
        return x, y, c, l, nbr_c, nbr_w, competencies, prev_interventions

    def _standardize_indices(self, intervention_idxs, batch_size):
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
                isinstance(intervention_idxs, torch.Tensor) and
                ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(f'Unsupported intervention indices {intervention_idxs}')
        if len(intervention_idxs.shape) == 1:
            intervention_idxs = torch.tile(torch.unsqueeze(intervention_idxs, 0), (batch_size, 1))
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead '
                f'we got indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
            f'Unsupported intervention indices with shape {intervention_idxs.shape}.'
        )
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs

    def _concept_intervention(
            self,
            c_pred,
            intervention_idxs=None,
            c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        c_pred_copy = c_pred.clone()
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
        )
        intervention_idxs = intervention_idxs.to(c_pred.device)
        if self.extra_dims:
            set_intervention_idxs = torch.nn.functional.pad(
                intervention_idxs,
                pad=(0, self.extra_dims),  # Just pads the last dimension
            )
        else:
            set_intervention_idxs = intervention_idxs
        if self.sigmoidal_prob:
            c_pred_copy[set_intervention_idxs] = c_true[intervention_idxs]
        else:
            active_intervention_values = self.active_intervention_values.to(c_pred.device)
            batched_active_intervention_values = torch.tile(
                torch.unsqueeze(active_intervention_values, 0),
                (c_pred.shape[0], 1)).to(c_true.device)
            inactive_intervention_values = self.inactive_intervention_values.to(c_pred.device)
            batched_inactive_intervention_values = torch.tile(
                torch.unsqueeze(inactive_intervention_values, 0),
                (c_pred.shape[0], 1)).to(c_true.device)
            c_pred_copy[set_intervention_idxs] = (
                    (c_true[intervention_idxs] * batched_active_intervention_values[intervention_idxs]) + (
                    (c_true[intervention_idxs] - 1) * -batched_inactive_intervention_values[intervention_idxs])
            )
        return c_pred_copy

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
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            latent = self.x2c_model(x)
        if self.sigmoidal_prob or self.bool:
            if self.extra_dims:
                c_pred_probs = self.sigmoid(latent[:, :-self.extra_dims])
                c_others = self.bottleneck_nonlin(latent[:, -self.extra_dims:])
                c_pred = torch.cat([c_pred_probs, c_others], dim=-1)
                c_sem = c_pred_probs
            else:
                c_pred = self.sigmoid(latent)
                c_sem = c_pred
        else:
            c_pred = latent
            if self.extra_dims:
                c_sem = self.sigmoid(latent[:, :-self.extra_dims])
            else:
                c_sem = self.sigmoid(latent)
        pos_embeddings = torch.ones(c_sem.shape).to(x.device)
        neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
        if output_embeddings or (intervention_idxs is None) and (c is not None) and (
                self.intervention_policy is not None) and not (self.sigmoidal_prob or self.bool):
            if (self.active_intervention_values is not None) and (self.inactive_intervention_values is not None):
                active_intervention_values = self.active_intervention_values.to(c_pred.device)
                pos_embeddings = torch.tile(active_intervention_values, (c.shape[0], 1)
                                            ).to(active_intervention_values.device)
                inactive_intervention_values = self.inactive_intervention_values.to(c_pred.device)
                neg_embeddings = torch.tile(inactive_intervention_values, (c.shape[0], 1)
                                            ).to(inactive_intervention_values.device)
            else:
                out_embs = c_pred.detach().cpu().numpy()
                for concept_idx in range(self.n_concepts):
                    pos_embeddings[:, concept_idx] = np.percentile(out_embs[:, concept_idx], 95)
                    neg_embeddings[:, concept_idx] = np.percentile(out_embs[:, concept_idx], 5)
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        if (intervention_idxs is None) and (c is not None) and (self.intervention_policy is not None):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=None
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        if self.bool:
            y = self.c2y_model((c_pred > 0.5).float())
        else:
            y = self.c2y_model(c_pred)

        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        return tuple([c_sem, c_pred, y] + tail_results)

    def forward(
            self,
            x,
            c=None,
            y=None,
            l=None,
            latent=None,
            intervention_idxs=None,
            competencies=None,
            prev_interventions=None,
            output_embeddings=False,
            output_latent=None,
            output_interventions=None
    ):
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            l=l,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
            output_embeddings=output_embeddings,
            output_latent=output_latent,
            output_interventions=output_interventions
        )

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
        )

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
        c_pseudo = c_pseudo.float()

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
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]

        if self.task_loss_weight != 0:
            task_loss = self.loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = 0
            task_loss_scalar = 0
        if self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(c_sem[l], c[l])
            concept_loss += self.loss_concept(c_sem[~l], c_pseudo[~l])  # TODO: unlabeled data using pseudo label
            concept_loss_scalar = concept_loss.detach()
            loss = self.concept_loss_weight * concept_loss + task_loss
        else:
            loss = task_loss
            concept_loss_scalar = 0.0
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(c_sem, y_logits, c, y)
        result = {
            "c_acc": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
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
                "concept_loss": result['concept_loss'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
            },
        }

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = ("auc" in name)
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

    def plot_heatmap(
            self,
            x,
            x_show=None,
            c=None,
            y=None,
            output_dir='',
    ):
        """
        Implemented in CEM
        :return: None
        """
        pass
