import os
import sys

sys.path.append(os.environ['RAVENS_ROOT'])
sys.path.append(os.environ['PLANNER'])
sys.path.append(os.path.join(os.environ['PLANNER'], '../'))

from data_utils import latest_file_from_dir
from dataset import DataModule
import planLM
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging


@hydra.main(config_path=os.path.join(os.environ['RAVENS_ROOT'], 'ravens/cfg'),
            config_name='train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg['wandb']['run_name']) if cfg['train']['log'] else None

    # Checkpoint saver
    # hydra_dir = Path(os.getcwd())
    checkpoint_path = cfg['train']['ckpt_dir']
    # last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    # last_checkpoint = \
    #     last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=cfg['wandb']['saver']['monitor'],
        save_top_k=2,
        save_last=True  # not necessarily the best model
    )
    # print(checkpoint_callback.best_model_path)

    # trainer
    max_epochs = cfg['train']['max_epochs']
    # limit_train_batches=10
    # limit_val_batches=10
    # sync_batchnorm=True  for torch.no_grad - use it for inference
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=max_epochs // 10,
        # resume_from_checkpoint=last_checkpoint,
        logger=wandb_logger,
        accelerator='gpu',
        devices="auto",  # cfg['train']['gpu']
        strategy="ddp_find_unused_parameters_false",
        callbacks=[checkpoint_callback]
        # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), checkpoint_callback]
    )

    # dataset
    # Config
    train = cfg['train']
    task = cfg['train'].task
    completed_actions = {'put-block-in-bowl': 'done placing blocks in bowls',
                         'towers-of-hanoi-seq': 'done putting disks in rods'}
    # name = '{}-{}'.format(task, model)

    # get train file path
    input_path = latest_file_from_dir(train.input_path)
    # dataset and dataloader in Lightning Module
    datamodule = \
        DataModule(train.model, input_path,
                   train.preprocess_path, train.save_path,
                   train.batch_size, train.train_val_split,
                   completed_action=completed_actions[task])

    # training
    print(f"\n==================== Training: {train.model} ========================\n "
          f"                    {train.task}                             \n"
          "=================================================================\n")
    pl_model = planLM.names[train.model](train.lr, train.weight_decay)
    trainer.fit(pl_model, datamodule)

    # tensorboard --logdir lightning_logs/


if __name__ == '__main__':
    main()
