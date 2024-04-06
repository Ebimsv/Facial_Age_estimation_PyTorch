import torch.cuda

config = {
    'img_width': 128,
    'img_height': 128,
    'img_size': 128,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model_name': 'resnet',
    'root_dir': '',
    'csv_path': '',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_path_test': '/home/deep/projects/Mousavi/Facial_Age_estimation_PyTorch/img_test/30_1_2.jpg',
    'output_path_test': '/home/deep/projects/Mousavi/Facial_Age_estimation_PyTorch/img_test/output.jpg',
    'leaky_relu': False,
    'epochs': 2,
    'batch_size': 128,
    'eval_batch_size': 256,
    'seed': 42,
    'lr': 0.0001,
    'wd': 0.001,
    'save_interval': 1,
    'reload_checkpoint': None,
    'finetune': 'weights/FA_DOCS/crnn-fa-base.pt',
    # 'finetune': None,
    'weights_dir': 'weights',
    'log_dir': 'logs',
    'cpu_workers': 4,
}
