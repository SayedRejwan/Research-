CONFIG = {
    # Data
    "data_path": "F:\\MLMI_SPLIT",
    "img_size": 224,
    "batch_size": 32,
    "augment": True,

    # Training
    "epochs": 40,
    "lr": 1e-4,
    "optimizer": "adam",  # adam | sgd
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,

    # Architecture
    "use_batchnorm": True,
    "dropout": 0.3,

    # Evaluation
    "use_gradcam": True,
    "output_dir": "experiments/run_1",
}
