import yaml
import time
import zipfile
from utils import *
from config import *

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    print(f"Using device: {args.device}")
    # setup_seed(args.seed)  # make the experiment repeatable
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    # ==================================================================================================
    # 2. Save codes and settings
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Save codes and settings ' + '=' * 36)
    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    with open(os.path.join(save_dir, 'args.yml'), 'a') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # ==================================================================================================
    # 3. Prepare data
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Prepare data ' + '=' * 36)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Build models ' + '=' * 36)

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
