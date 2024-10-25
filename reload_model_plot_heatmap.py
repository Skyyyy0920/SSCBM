import pylab as pl
import yaml
from collections import defaultdict
from utils_ori import *
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
from data.cub_loader import CONCEPT_SEMANTICS, SELECTED_CONCEPTS
import utils.factory as factory


class CUBDataset_for_heatmap(Dataset):
    def __init__(self, pkl_file_paths, image_dir,
                 root_dir='/scratch/xg02913/CUB_200_2011', path_transform=None, transform=None,
                 concept_transform=None, label_transform=None):
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            with open(file_path, 'rb') as f:
                self.data.extend(pickle.load(f))
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.path_transform = path_transform
        self.model_try, _, self.CLIP_preprocess = factory.create_model_and_transforms('ViT-B/32')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        img_path = img_path.replace(
            '/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/',
            '/scratch/xg02913/CUB_200_2011/'
        )
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224,224))
        img = self.CLIP_preprocess(img)
        

        match = re.search(r'CUB_200_2011/images/(.*?).jpg', img_path)
        if match:
            intermediate_str = match.group(1)
            final_str = intermediate_str.split('/')[-1]
        else:
            final_str = 'oo'

        transform = transforms.Compose([
            transforms.CenterCrop(224),
            #transforms.ToTensor(),  # implicitly divides by 255
        ])
        img_show = transform(img)

        class_label = img_data['class_label']
        if self.label_transform:
            class_label = self.label_transform(class_label)
        if self.transform:
            img = self.transform(img)

        attr_label = img_data['attribute_label']
        if self.concept_transform is not None:
            attr_label = self.concept_transform(attr_label)

        return img, img_show, class_label, torch.FloatTensor(attr_label), final_str


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
    print('++++++++start++++++')
    pl.seed_everything(20010125)
    logging.info(f"Reload the trained model to plot the heatmap!")
    args = get_args()
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    with open(f"configs/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    experiment_config["model_pretrain_path"] = "/scratch/xg02913/SSCBM/checkpoints/CUB-200-2011_20-42/test.pt"

    dataset_config = experiment_config['dataset_config']
    if args.dataset == "CUB-200-2011":
        data_module = cub_data_module
    elif args.dataset == "CelebA":
        data_module = celeba_data_module
    elif args.dataset == "MNIST":
        data_module = mnist_data_module
    elif args.dataset in ["XOR", "vector", "Dot", "Trigonometric"]:
        data_module = get_synthetic_data(dataset_config["dataset"])
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")
    print('======================= dataset config ok ==========================')
    if experiment_config['c_extractor_arch'] == "mnist_extractor":
        num_operands = dataset_config.get('num_operands', 32)
        experiment_config["c_extractor_arch"] = mnist_data_module.get_mnist_extractor_arch(
            input_shape=(dataset_config.get('batch_size', 512), num_operands, 28, 28),
            num_operands=num_operands,
        )
    elif experiment_config['c_extractor_arch'] == 'synth_extractor':
        input_features = get_synthetic_num_features(dataset_config["dataset"])
        experiment_config["c_extractor_arch"] = get_synthetic_extractor_arch(input_features)
    print('======================= start generate data ==========================')
    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = data_module.generate_data(
        config=dataset_config,
        seed=20010125,
        labeled_ratio=args.labeled_ratio,
    )
    logging.info(f"imbalance: {imbalance}")
    print('======================= get imbalance ==========================')
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
            print('start load model')
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
                seed=429,
                imbalance=imbalance,
                gradient_clip_val=run_config.get('gradient_clip_val', 0),
            )
            print('get model and model result')
            transform = transforms.Compose([
                transforms.CenterCrop(224),
                #transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
            ])

            root_dir = '/scratch/xg02913/CUB_200_2011'
            base_dir = os.path.join(root_dir, 'class_attr_data_10')
            train_data_path = os.path.join(base_dir, 'train.pkl')
            val_data_path = os.path.join(base_dir, 'val.pkl')
            test_data_path = os.path.join(base_dir, 'test.pkl')
            print('make dataset')
            dataset = CUBDataset_for_heatmap(
                pkl_file_paths=[train_data_path],
                image_dir='images',
                transform=transform,
                root_dir=root_dir,
            )
            print('start loading data.......')
            loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=64)
            print('finish loading data.......')
            concept_set = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
            print('finish generate concept set.......')
            for b_idx, batch in enumerate(loader):
                print('loader...')
                x, x_show, y, c, img_name = batch
                print('try to plot heatmap')
                model.plot_heatmap(x, x_show, c, y, img_name, f"{save_dir}/heatmap", concept_set)
                break

    print(f"========================finish========================")
