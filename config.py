class Config:
    DISTRIBUTED = "model_parallel"
    EPOCHS = 50
    DATA_ROOT = "./data"
    BATCH_SIZE = 64
    LOG_FILE = "./logs/" + DISTRIBUTED + "_train_log"
    SAVE_MODEL_PATH = "./models/" + DISTRIBUTED
    SAVE_LOSS_PATH = "./plot/" + DISTRIBUTED + "/loss.png"
    SAVE_ACCURACY_PATH = "./plot/" + DISTRIBUTED + "/accuracy.png"
    ENCODER_LEARNING_RATE = 1e-5
    HEAD_LEARNING_RATE = 1e-3