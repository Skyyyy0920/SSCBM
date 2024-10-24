import clip
from torch import nn
import pytorch_lightning as pl
from torchvision.models import resnet50
from models.cbm import ConceptBottleneckModel
import train.utils as utils
from utils import *
from prs_hook import hook_prs_logger


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            shared_prob_gen=True,
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
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation is None:
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                    ])
                )
            elif embedding_activation == "sigmoid":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.Sigmoid(),
                    ])
                )
            elif embedding_activation == "leakyrelu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.LeakyReLU(),
                    ])
                )
            elif embedding_activation == "relu":
                self.concept_context_generators.append(
                    torch.nn.Sequential(*[
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            2 * emb_size,
                        ),
                        torch.nn.ReLU(),
                    ])
                )
            if self.shared_prob_gen and len(self.concept_prob_generators) == 0:
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(2 * emb_size, 1))
        if c2y_model is None:
            # Else we construct it here directly
            units = [
                        n_concepts * emb_size
                    ] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_concept_labeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_concept_unlabeled = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
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

        self.fc = nn.Linear(512, self.emb_size)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.CLIP, self.CLIP_preprocess = clip.load('ViT-B/32', self.device)
        self.prs = hook_prs_logger(self.CLIP, self.device)

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

    def unlabeled_image_encoder(self, x):
        # self.pre_concept_model resnet34
        x = self.pre_concept_model.conv1(x)
        x = self.pre_concept_model.bn1(x)
        x = self.pre_concept_model.relu(x)
        x = self.pre_concept_model.maxpool(x)

        x = self.pre_concept_model.layer1(x)
        x = self.pre_concept_model.layer2(x)
        x = self.pre_concept_model.layer3(x)
        x = self.pre_concept_model.layer4(x)
        x = x.transpose(1, 3)
        x = self.fc(x)
        return x

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
            pre_c = self.pre_concept_model(x)  # [batch_size, 299, 299] -> [batch_size, resnet_out_features]
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)  # [batch_size, resnet_out_features] -> [batch_size, 2 * emb_size]
                prob = prob_gen(context)  # [batch_size, 2 * emb_size] -> [batch_size, 1]
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sigmoid(prob))
            c_sem = torch.cat(c_sem, axis=-1)  # [batch_size, 1, n_concepts] -> [batch_size, n_concepts]
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
                self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, :self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size:],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the negative embeddings
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred = c_embedding.view((-1, self.emb_size * self.n_concepts))
        y_pred = self.c2y_model(c_pred)

        with torch.no_grad():
            image_features = self.CLIP.encode_image(x)  # [batch, 3, 224, 224] -> [batch, 512]
        image_features /= image_features.norm(dim=-1, keepdim=True)

        c_pred_unlabeled = []
        for i in range(len(y)):
            text_inputs = []
            for c in cub_data_module.CONCEPT_MAP_P:
                text_inputs.append(clip.tokenize(f"a photo of a {cub_data_module.CLASS_NAMES[y[i]]} ({c})"))
            text_inputs = torch.cat(text_inputs).to(x.device)
            with torch.no_grad():
                text_features = self.CLIP.encode_text(text_inputs)

            text_features /= text_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # TODO
            # c_pred_unlabeled = self.sigmoid(similarity)
            similarity = image_features[i] @ text_features.T
            c_pred_unlabeled.append(similarity)

        c_pred_unlabeled = torch.stack(c_pred_unlabeled)

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

        return tuple([c_sem, c_pred, c_pred_unlabeled, y_pred] + tail_results)

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
        pre_c = self.pre_concept_model(x)
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
        c_sem = torch.cat(c_sem, axis=-1)
        contexts = torch.cat(contexts, axis=1)

        probs = c_sem
        c_embedding = (
                contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
                contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        # image_feature = self.unlabeled_image_encoder(x)
        
        prs.reinit()
        with torch.no_grad():
            image_features = self.CLIP.encode_image(x)  # [batch, 3, 224, 224] -> [batch, 512]
            attentions, mlps = prs.finalize(image_features)
        # image_features /= image_features.norm(dim=-1, keepdim=True)

        all_features = []

        for i in range(len(y)):
            text_inputs = []
            for c in cub_data_module.CONCEPT_MAP_P:
                text_inputs.append(clip.tokenize(f"a photo of a {cub_data_module.CLASS_NAMES[y[i]]} ({c})"))
            text_inputs = torch.cat(text_inputs).to(x.device)
            with torch.no_grad():
                text_features = self.CLIP.encode_text(text_inputs)

            text_features /= text_features.norm(dim=-1, keepdim=True)
            #all_text_features.append(text_features)  ## need action    
            attention_map = attentions[i, :, 1:, :].sum(axis=(0,2)) @ text_features.T ## need action
            all_features.append(attention_map)
        
        # heatmap = []
        # for i in range(len(image_feature)):
        #     heatmap.append(torch.matmul(image_feature[i], c_embedding[i].transpose(0, 1)))
        # heatmap = torch.stack(heatmap).permute(0, 3, 1, 2)

        for i in range(len(x)):
            save_dir = f"./{output_dir}/{img_name[i]}"
            
            # save_dir = f"/root/autodl-tmp/heatmap/{img_name[i]}"

            # visualize_and_save_heatmaps(x_show[i], c[i], c_sem[i], heatmap[i], save_dir, concept_set)
            visualize_and_save_attentionmaps(x_show[i], c[i], c_sem[i], all_features[i], save_dir, concept_set)
