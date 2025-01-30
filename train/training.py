import os
import time
import copy
import torch
import joblib
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, loggers

import cem.train.utils as utils
from models.construction import construct_model


def evaluate_cbm(
        model,
        trainer,
        config,
        run_name,
        old_results=None,
        rerun=False,
        test_dl=None,
        val_dl=None,
):
    eval_results = {}
    for (current_dl, dl_name) in [(val_dl, "val"), (test_dl, "test")]:
        logging.info(f"{dl_name}")
        model.freeze()

        def _inner_call():
            [eval_results] = trainer.test(model, current_dl)
            output = [
                eval_results[f"test_c_acc"],
                eval_results[f"test_y_acc"],
                eval_results[f"test_c_auc"],
                eval_results[f"test_y_auc"],
                eval_results[f"test_c_f1"],
                eval_results[f"test_y_f1"],
            ]
            top_k_vals = []
            for key, val in eval_results.items():
                if f"test_y_top" in key:
                    top_k = int(key[len(f"test_y_top_"):-len("_accuracy")])
                    top_k_vals.append((top_k, val))
            output += list(map(
                lambda x: x[1],
                sorted(top_k_vals, key=lambda x: x[0]),
            ))
            return output

        keys = [
            f"{dl_name}_acc_c",
            f"{dl_name}_acc_y",
            f"{dl_name}_auc_c",
            f"{dl_name}_auc_y",
            f"{dl_name}_f1_c",
            f"{dl_name}_f1_y",
        ]
        if 'top_k_accuracy' in config:
            top_k_args = config['top_k_accuracy']
            if top_k_args is None:
                top_k_args = []
            if not isinstance(top_k_args, list):
                top_k_args = [top_k_args]
            for top_k in sorted(top_k_args):
                keys.append(f'{dl_name}_top_{top_k}_acc_y')
        values, _ = utils.load_call(
            function=_inner_call,
            keys=keys,
            run_name=run_name,
            old_results=old_results,
            rerun=rerun,
            kwargs={},
        )
        eval_results.update({
            key: val
            for (key, val) in zip(keys, values)
        })
    return eval_results


def train_end_to_end_model(
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

    full_run_name = run_name

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
    # logging.info(f"{model}")
    logging.info(f"Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logging.info(f"Number of non-trainable parameters in model: "
                 f"{sum(p.numel() for p in model.parameters() if not p.requires_grad)}")

    if config.get("model_pretrain_path"):
        if os.path.exists(config.get("model_pretrain_path")):
            logging.info("Load pretrained model")
            model.load_state_dict(torch.load(config.get("model_pretrain_path")), strict=False)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config['max_epochs'],
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
        logger=logger or False,
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=gradient_clip_val,
    )

    if result_dir:
        if activation_freq:
            fit_trainer = utils.ActivationMonitorWrapper(
                model=model,
                trainer=trainer,
                activation_freq=activation_freq,
                single_frequency_epochs=single_frequency_epochs,
                output_dir=os.path.join(result_dir, f"test_embedding/{full_run_name}"),
                test_dl=val_dl,  # Pass the validation data intentionally to avoid explosion of memory usage
            )
        else:
            fit_trainer = trainer
    else:
        fit_trainer = trainer

    model_saved_path = os.path.join(result_dir or ".", f'{full_run_name}.pt')
    if not rerun and os.path.exists(model_saved_path):
        logging.info("Found cached model... loading it")
        model.load_state_dict(torch.load(model_saved_path))
        if os.path.exists(model_saved_path.replace(".pt", "_training_times.npy")):
            [training_time, num_epochs] = np.load(model_saved_path.replace(".pt", "_training_times.npy"))
        else:
            training_time, num_epochs = 0, 0
    else:
        start_time = time.time()
        fit_trainer.fit(model, train_dl, val_dl)
        training_time = time.time() - start_time
        num_epochs = fit_trainer.current_epoch
        if save_model and result_dir:
            torch.save(model.state_dict(), model_saved_path)
            np.save(model_saved_path.replace(".pt", "_training_times.npy"), np.array([training_time, num_epochs]))

    if not os.path.exists(os.path.join(result_dir, f'{run_name}_experiment_config.joblib')):
        # Then let's serialize the experiment config for this run
        config_copy = copy.deepcopy(config)
        if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
        ):
            del config_copy["c_extractor_arch"]
        joblib.dump(config_copy, os.path.join(result_dir, f'{run_name}_experiment_config.joblib'))
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
    eval_results['training_time'] = training_time
    eval_results['num_epochs'] = num_epochs
    if test_dl is not None:
        logging.info(f'c_acc: {eval_results["test_acc_c"] * 100:.2f}%')
        logging.info(f'y_acc: {eval_results["test_acc_y"] * 100:.2f}%')
        logging.info(f'c_auc: {eval_results["test_auc_c"] * 100:.2f}%')
        logging.info(f'y_auc: {eval_results["test_auc_y"] * 100:.2f}%')
        logging.info(f'with {num_epochs} epochs in {training_time / 60:.2f} minutes')

    return model, eval_results


def update_statistics(
        aggregate_results,
        run_config,
        test_results,
        run_name,
        model=None,
        prefix='',
):
    print(test_results)
    for key, val in test_results.items():
        aggregate_results[prefix + key] = val
