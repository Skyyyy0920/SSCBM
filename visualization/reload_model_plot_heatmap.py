import yaml
import zipfile
from collections import defaultdict
from utils import *
from train.evaluate import *
from configs.basic_config import *
from models.construction import construct_model
from train.training import evaluate_cbm


def load_evaluate_model(
        n_concepts,
        n_tasks,
        config,
        train_dl,
        val_dl,
        run_name,
        result_dir=None,
        test_dl=None,
        imbalance=None,
        task_class_weights=None,
        rerun=False,
        logger=False,
        project_name='',
        split=0,
        seed=42,
        save_model=True,
        activation_freq=0,
        single_frequency_epochs=0,
        gradient_clip_val=0,
        old_results=None,
        enable_checkpointing=False,
        accelerator="auto",
        devices="auto",
):
    seed_everything(seed)

    full_run_name = "test"

    logging.info(f"Training ***{run_name}***")
    for key, val in config.items():
        logging.info(f"{key} -> {val}")

    # create model
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
    logging.info(f"Reload the trained model to plot the heatmap!")
    args = get_args()
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    with open(f"configs/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    experiment_config["model_pretrain_path"] = "../saved_checkpoints/CUB-200-2011_12-32/test.pt"

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
    import time
    import random
    current_time = time.time()
    random.seed(current_time)
    if not train and self.output_image and self.current_epoch >= 50:
        logging_time = time.strftime('%H-%M-%S', time.localtime())
        save_dir = os.path.join(f"heatmap", f"{logging_time}")
        visualize_and_save_heatmaps(
            x_.detach().cpu(),
            heatmap.detach().cpu(),
            sample_index=random.randint(0, len(x_)),
            output_dir=save_dir,
            data_save_path='saved_data.pth'
        )
        self.output_image = False

    print(f"========================finish========================")
