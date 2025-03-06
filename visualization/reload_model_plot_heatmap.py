import pylab as pl
import yaml
from collections import defaultdict
from utils import *
from train.evaluate import *
from configs.basic_config import *
from models.construction import construct_model
from train.training import evaluate_cbm
import os
import torch
import pickle
import logging
from pytorch_lightning import seed_everything
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from data.cub_loader import CONCEPT_SEMANTICS, SELECTED_CONCEPTS, CUBDataset_for_heatmap


def load_evaluate_model(
        n_concepts,
        n_tasks,
        config,
        train_dl,
        val_dl,
        run_name,
        test_dl=None,
        imbalance=None,
        task_class_weights=None,
        rerun=False,
        logger=False,
        seed=42,
        gradient_clip_val=0,
        old_results=None,
        enable_checkpointing=False,
        accelerator="auto",
        devices="auto",
):
    seed_everything(seed)
    model = construct_model(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )

    if config.get("model_pretrain_path"):
        if os.path.exists(config.get("model_pretrain_path")):
            logging.info("Load pretrained model")
            model.load_state_dict(torch.load(config.get("model_pretrain_path")), strict=False)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config['max_epochs'],
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
        # callbacks=callbacks,
        logger=logger or False,
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=gradient_clip_val,
    )

    eval_results = evaluate_cbm(
        model=model,
        trainer=trainer,
        config=config,
        run_name=run_name,
        old_results=old_results,
        rerun=rerun,
        test_dl=test_dl,
        val_dl=val_dl,
    )

    if test_dl is not None:
        logging.info(f'c_acc: {eval_results["test_acc_c"] * 100:.2f}%')
        logging.info(f'y_acc: {eval_results["test_acc_y"] * 100:.2f}%')
        logging.info(f'c_auc: {eval_results["test_auc_c"] * 100:.2f}%')
        logging.info(f'y_auc: {eval_results["test_auc_y"] * 100:.2f}%')

    return model, eval_results


if __name__ == '__main__':
    pl.seed_everything()
    logging.info(f"Reload the trained model to plot the heatmap!")
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
    elif args.dataset == "CelebA":
        data_module = celeba_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = data_module.generate_data(
        config=dataset_config,
        seed=20010125,
        labeled_ratio=args.labeled_ratio,
    )
    logging.info(f"imbalance: {imbalance}")

    task_class_weights = update_config_with_dataset(
        config=experiment_config,
        train_dl=train_dl,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_map=concept_map,
    )

    results = defaultdict(dict)
    for current_config in experiment_config['runs']:
        run_name = current_config['architecture']
        trial_config = copy.deepcopy(experiment_config)
        trial_config.update(current_config)

        for run_config in generate_hyper_param_configs(trial_config):
            run_config = copy.deepcopy(run_config)
            run_config['result_dir'] = save_dir
            evaluate_expressions(run_config, soft=True)

            model, model_results = load_evaluate_model(
                run_name=run_name,
                task_class_weights=task_class_weights,
                accelerator=args.device,
                devices='auto',
                n_concepts=run_config['n_concepts'],
                n_tasks=run_config['n_tasks'],
                config=run_config,
                train_dl=train_dl,
                val_dl=val_dl,
                test_dl=test_dl,
                seed=703,
                imbalance=imbalance,
                gradient_clip_val=run_config.get('gradient_clip_val', 0),
            )

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

            loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=64)
            concept_set = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
            for b_idx, batch in enumerate(loader):
                x, x_show, y, c, img_name = batch
                model.plot_heatmap(x, x_show, c, y, img_name, f"{save_dir}/heatmap", concept_set)
                break

    print(f"========================finish========================")
