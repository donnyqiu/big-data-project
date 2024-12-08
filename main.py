from config import Config
from train import Trainer
import torch
import torch.multiprocessing as mp

def main(rank, world_size):
    Trainer(
        Config.DISTRIBUTED,
        rank,
        world_size,
        Config.EPOCHS,
        Config.BATCH_SIZE,
        Config.ENCODER_LEARNING_RATE,
        Config.HEAD_LEARNING_RATE,
        Config.DATA_ROOT,
        Config.SAVE_MODEL_PATH,
        Config.LOG_FILE,
        Config.SAVE_LOSS_PATH,
        Config.SAVE_ACCURACY_PATH
    ).train()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if Config.DISTRIBUTED == "data_parallel":
        mp.spawn(
            main,
            args=(world_size,),
            nprocs=world_size,
            join=True)
    else:
        main(-1, world_size)
