import yaml
import zipfile
from collections import defaultdict

from utils import *
from train.training import *
import cem.train.evaluate as evaluation
from configs.basic_config import *
import data.cub_loader as cub_data_module
import data.mnist_loader as mnist_data_module
import data.celeba_loader as celeba_data_module
from data.synthetic_loader import get_synthetic_data, get_synthetic_num_features, get_synthetic_extractor_arch

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    args = get_args()
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    with open(f"configs/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # ==================================================================================================
    # 2. Save codes and settings
    # ==================================================================================================
    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    with open(os.path.join(save_dir, 'args.yml'), 'a') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    with open(os.path.join(save_dir, f"experiment_config.yaml"), "w") as f:
        yaml.dump(experiment_config, f)

    # ==================================================================================================
    # 3. Prepare data
    # ==================================================================================================
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

    if experiment_config['c_extractor_arch'] == "mnist_extractor":
        num_operands = dataset_config.get('num_operands', 32)
        experiment_config["c_extractor_arch"] = mnist_data_module.get_mnist_extractor_arch(
            input_shape=(dataset_config.get('batch_size', 512), num_operands, 28, 28),
            num_operands=num_operands,
        )
    elif experiment_config['c_extractor_arch'] == 'synth_extractor':
        input_features = get_synthetic_num_features(dataset_config["dataset"])
        experiment_config["c_extractor_arch"] = get_synthetic_extractor_arch(input_features)

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = data_module.generate_data(
        config=dataset_config,
        seed=42,
        labeled_ratio=experiment_config['labeled_ratio'],
    )

    task_class_weights = update_config_with_dataset(
        config=experiment_config,
        train_dl=train_dl,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_map=concept_map,
    )

    # ==================================================================================================
    # 4. Build models, define overall loss and optimizer. Then training
    # ==================================================================================================
    results = defaultdict(dict)
    for current_config in experiment_config['runs']:
        run_name = current_config['architecture']
        trial_config = copy.deepcopy(experiment_config)
        trial_config.update(current_config)

        for run_config in generate_hyper_param_configs(trial_config):
            run_config = copy.deepcopy(run_config)
            run_config['result_dir'] = save_dir
            evaluate_expressions(run_config, soft=True)

            old_results = None

            model, model_results = train_end_to_end_model(
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
                split=0,
                result_dir=save_dir,
                project_name=args.project_name,
                seed=42,
                imbalance=imbalance,
                old_results=old_results,
                gradient_clip_val=run_config.get('gradient_clip_val', 0),
                activation_freq=args.activation_freq,
                single_frequency_epochs=args.single_frequency_epochs,
            )

            update_statistics(
                aggregate_results=results[run_name],
                run_config=run_config,
                model=model,
                test_results=evaluation.evaluate_representation_metrics(
                    config=run_config,
                    n_concepts=run_config['n_concepts'],
                    n_tasks=run_config['n_tasks'],
                    test_dl=test_dl,
                    run_name=run_name,
                    imbalance=imbalance,
                    result_dir=save_dir,
                    task_class_weights=task_class_weights,
                    accelerator=args.device,
                    devices='auto',
                    seed=42,
                    old_results=old_results,
                ),
                run_name=run_name,
                prefix="",
            )
            results[run_name][f'num_trainable_params'] = \
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            results[run_name][f'num_non_trainable_params'] = \
                sum(p.numel() for p in model.parameters() if not p.requires_grad)

    with open(f'{save_dir}/results.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

