# config.py
cfg_mnet = {
    'name': 'PLNet',
    'min_sizes': [[12, 24], [48, 96], [192, 320]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 256,
    'ngpu': 3,
    'epoch': 30,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_slim = {
    'name': 'slim_224_rgb',
    # 'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    # 'steps': [8, 16, 32, 64],
    'min_sizes': [[12, 24], [48, 96], [192, 320]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size':  512,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 10,
    'decay2': 15,
    'image_size': 224
}

cfg_rfb = {
    'name': 'RFB',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300
}


