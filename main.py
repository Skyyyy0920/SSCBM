import yaml
import time
import zipfile
from utils import *
from config.basic_config import *
import data.cub_loader as cub_data_module

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment configuration ' + '=' * 36)
    args = get_args()
    # setup_seed(args.seed)  # make the experiment repeatable
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    with open(f"./config/{args.dataset}.yaml", "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    # ==================================================================================================
    # 2. Save codes and settings
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Save codes and settings ' + '=' * 36)
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
    print('\n' + '=' * 36 + ' Prepare data ' + '=' * 36)
    dataset_config = experiment_config['dataset_config']
    if args.dataset == "CUB-200-2011":
        data_module = cub_data_module
    else:
        raise ValueError(f"Unsupported dataset {dataset_config['dataset']}!")

    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = data_module.generate_data(
        config=dataset_config,
        seed=42,
        output_dataset_vars=True)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Build models ' + '=' * 36)

    import pytorch_lightning as pl
    from cem.models.cem import ConceptEmbeddingModel
    cem_model = ConceptEmbeddingModel(
        n_concepts=n_concepts,  # Number of training-time concepts
        n_tasks=n_tasks,  # Number of output labels
        emb_size=16,
        concept_loss_weight=0.1,
        learning_rate=1e-3,
        optimizer="adam",
        training_intervention_prob=0.25,  # RandInt probability
    )

    #####
    # Train it
    #####

    trainer = pl.Trainer(
        accelerator="gpu",  # or "cpu" if no GPU available
        devices="auto",
        max_epochs=100,
        check_val_every_n_epoch=5,
    )
    # train_dl and val_dl are datasets previously built...
    trainer.fit(cem_model, train_dl, val_dl)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================

    # ==================================================================================================
    # 7. Training
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Start training ' + '=' * 36)

    # ==================================================================================================
    # 8. Validation and Testing
    # ==================================================================================================
