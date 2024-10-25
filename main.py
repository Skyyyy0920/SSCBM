import yaml
import zipfile
from collections import defaultdict

from utils_ori import *
from train.training import *
from train.evaluate import *
from configs.basic_config import *

import interventions.utils as intervention_utils

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
    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
        task_class_weights,
        acquisition_costs
    ) = generate_dataset_and_update_config(experiment_config, args)

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
                seed=args.seed,
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
                test_results=evaluate_representation_metrics(
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
                    seed=args.seed,
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
            for key, value in results[run_name].items():
                f.write(f"{key}: {value}\n")

    print(f"========================finish========================")
