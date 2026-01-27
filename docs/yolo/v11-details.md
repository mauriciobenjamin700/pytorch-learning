# Investigando A YOLOV11 com Profundidade

Modelo YOLO Pré-Treinado

```json
{'date': '2025-10-27T14:43:45.791915', 'version': '8.3.218', 'license': 'AGPL-3.0 (https://ultralytics.com/license)', 'docs': 'https://docs.ultralytics.com', 'epoch': -1, 'best_fitness': None, 'model': ClassificationModel(
  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (1): Conv(
      (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (2): C3k2(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(48, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): ModuleList(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (4): C3k2(
      (cv1): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): ModuleList(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (6): C3k2(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): ModuleList(
        (0): C3k(
          (cv1): Conv(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (8): C3k2(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): ModuleList(
        (0): C3k(
          (cv1): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv2): Conv(
            (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (cv3): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU()
          )
          (m): Sequential(
            (0): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
            (1): Bottleneck(
              (cv1): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
              (cv2): Conv(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): SiLU()
              )
            )
          )
        )
      )
    )
    (9): C2PSA(
      (cv1): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (m): Sequential(
        (0): PSABlock(
          (attn): Attention(
            (qkv): Conv(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity()
            )
            (proj): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity()
            )
            (pe): Conv(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (ffn): Sequential(
            (0): Conv(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): SiLU()
            )
            (1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
        )
      )
    )
    (10): Classify(
      (conv): Conv(
        (conv): Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (pool): AdaptiveAvgPool2d(output_size=1)
      (drop): Dropout(p=0.0, inplace=True)
      (linear): Linear(in_features=1280, out_features=2, bias=True)
    )
  )
), 'ema': None, 'updates': None, 'optimizer': None, 'scaler': None, 'train_args': {'task': 'classify', 'mode': 'train', 'model': 'yolo11n-cls.pt', 'data': '/home/mauriciobenjamin700/projects/companies/iagro/famachapp-models/class-data', 'epochs': 100, 'time': None, 'patience': 100, 'batch': 16, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': '0', 'workers': 8, 'project': 'classify', 'name': 'class', 'exist_ok': False, 'pretrained': True, 'optimizer': 'auto', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'amp': True, 'fraction': 1.0, 'profile': False, 'freeze': None, 'multi_scale': False, 'compile': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': True, 'opset': None, 'workspace': None, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.0, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0, 'mixup': 0.0, 'cutmix': 0.0, 'copy_paste': 0.0, 'copy_paste_mode': 'flip', 'auto_augment': 'randaugment', 'erasing': 0.4, 'cfg': None, 'tracker': 'botsort.yaml'}, 'train_metrics': {'metrics/accuracy_top1': 0.85714, 'metrics/accuracy_top5': 1.0, 'val/loss': 0.5144, 'fitness': 0.92857}, 'train_results': {'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], 'time': [8.10166, 9.92182, 11.5775, 13.2376, 14.9222, 16.6324, 18.3226, 22.9279, 24.5762, 26.2308, 27.9285, 29.6506, 31.3091, 32.9788, 34.6279, 36.3304, 38.0187, 39.7619, 41.5168, 43.2715, 45.0595, 46.8034, 48.51, 50.2428, 52.0853, 56.8017, 58.6018, 60.3958, 62.1492, 63.9485, 65.856, 67.6893, 69.4716, 71.1712, 72.8264, 74.5526, 76.2848, 77.9678, 79.6656, 81.3986, 83.0774, 84.9991, 87.1057, 92.023, 94.0411, 95.8402, 97.6397, 99.4287, 101.179, 103.002, 104.831, 106.617, 108.4, 110.232, 111.998, 113.8, 115.598, 117.458, 119.334, 121.223, 123.026, 127.836, 129.723, 131.575, 133.606, 135.417, 137.278, 139.161, 141.045, 142.853, 144.72, 146.515, 148.28, 150.104, 151.867, 153.665, 155.434, 157.21, 159.013, 163.644, 165.489, 167.298, 168.95, 170.752, 172.479, 174.242, 175.931, 177.739, 179.5, 181.238, 183.723, 185.642, 187.469, 189.244, 190.975, 192.698, 197.437, 199.193, 200.936, 202.719], 'train/loss': [0.65876, 0.56803, 0.56134, 0.57017, 0.56092, 0.5593, 0.53155, 0.51932, 0.55453, 0.48603, 0.5203, 0.50921, 0.54296, 0.5071, 0.51903, 0.53351, 0.50838, 0.48553, 0.4886, 0.49044, 0.49135, 0.48966, 0.4803, 0.50416, 0.47742, 0.49854, 0.48758, 0.51568, 0.50753, 0.47915, 0.52454, 0.49738, 0.47073, 0.4852, 0.46962, 0.46004, 0.46463, 0.47849, 0.45877, 0.49085, 0.48069, 0.46725, 0.45024, 0.46808, 0.4339, 0.45667, 0.45631, 0.44267, 0.44302, 0.46131, 0.47287, 0.44492, 0.43654, 0.44144, 0.40771, 0.41281, 0.44547, 0.40798, 0.45233, 0.43625, 0.46017, 0.42665, 0.42264, 0.42426, 0.42105, 0.39864, 0.41139, 0.37449, 0.40573, 0.42057, 0.38322, 0.42181, 0.39594, 0.36881, 0.41666, 0.40156, 0.36164, 0.38089, 0.40611, 0.36163, 0.38312, 0.37449, 0.38702, 0.35908, 0.36727, 0.36395, 0.34551, 0.37111, 0.38327, 0.36443, 0.3454, 0.34621, 0.34789, 0.33793, 0.34858, 0.33785, 0.358, 0.35232, 0.33558, 0.34351], 'metrics/accuracy_top1': [0.70238, 0.75, 0.75, 0.65476, 0.67857, 0.71429, 0.67857, 0.7619, 0.71429, 0.7619, 0.80952, 0.72619, 0.77381, 0.75, 0.7619, 0.71429, 0.80952, 0.79762, 0.75, 0.72619, 0.78571, 0.72619, 0.7381, 0.69048, 0.7381, 0.70238, 0.75, 0.71429, 0.7619, 0.78571, 0.7381, 0.82143, 0.7381, 0.75, 0.79762, 0.7381, 0.78571, 0.7381, 0.71429, 0.7619, 0.75, 0.72619, 0.7381, 0.78571, 0.78571, 0.78571, 0.7619, 0.79762, 0.80952, 0.7619, 0.7619, 0.78571, 0.80952, 0.78571, 0.7381, 0.83333, 0.7619, 0.78571, 0.75, 0.77381, 0.77381, 0.79762, 0.79762, 0.83333, 0.79762, 0.79762, 0.78571, 0.82143, 0.7619, 0.78571, 0.77381, 0.78571, 0.80952, 0.77381, 0.80952, 0.83333, 0.80952, 0.82143, 0.80952, 0.80952, 0.78571, 0.77381, 0.79762, 0.80952, 0.82143, 0.85714, 0.82143, 0.79762, 0.78571, 0.83333, 0.83333, 0.80952, 0.82143, 0.79762, 0.83333, 0.80952, 0.79762, 0.77381, 0.79762, 0.78571], 'metrics/accuracy_top5': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'val/loss': [0.59326, 0.48193, 0.52987, 0.72282, 0.74479, 0.51799, 0.59163, 0.51318, 0.55542, 0.45833, 0.46753, 0.56738, 0.56193, 0.63582, 0.47192, 0.54215, 0.41162, 0.43701, 0.44751, 0.59082, 0.50505, 0.50171, 0.48429, 0.50488, 0.58309, 0.62956, 0.54077, 0.8916, 0.45662, 0.42928, 0.46444, 0.42708, 0.46395, 0.50627, 0.43188, 0.50993, 0.43481, 0.48006, 0.47298, 0.43221, 0.434, 0.41512, 0.5201, 0.45003, 0.45736, 0.46875, 0.43237, 0.45923, 0.4707, 0.44466, 0.46086, 0.43091, 0.44515, 0.56681, 0.46875, 0.41219, 0.43034, 0.40454, 0.51562, 0.44645, 0.45662, 0.40405, 0.40666, 0.40479, 0.40373, 0.40975, 0.45174, 0.38875, 0.45638, 0.39225, 0.47184, 0.41854, 0.39079, 0.43132, 0.38428, 0.37598, 0.45353, 0.40487, 0.4056, 0.41984, 0.44661, 0.41463, 0.45247, 0.46501, 0.49585, 0.5144, 0.47868, 0.50854, 0.49618, 0.49463, 0.49658, 0.56641, 0.54492, 0.49626, 0.54004, 0.56226, 0.51432, 0.54696, 0.5887, 0.55168], 'lr/pg0': [0.000542744, 0.00108754, 0.00162133, 0.00161749, 0.00160099, 0.00158448, 0.00156798, 0.00155148, 0.00153497, 0.00151847, 0.00150197, 0.00148546, 0.00146896, 0.00145246, 0.00143595, 0.00141945, 0.00140295, 0.00138644, 0.00136994, 0.00135344, 0.00133693, 0.00132043, 0.00130393, 0.00128742, 0.00127092, 0.00125442, 0.00123791, 0.00122141, 0.00120491, 0.0011884, 0.0011719, 0.0011554, 0.00113889, 0.00112239, 0.00110589, 0.00108938, 0.00107288, 0.00105638, 0.00103987, 0.00102337, 0.00100687, 0.000990365, 0.000973861, 0.000957358, 0.000940855, 0.000924352, 0.000907848, 0.000891345, 0.000874842, 0.000858338, 0.000841835, 0.000825332, 0.000808828, 0.000792325, 0.000775822, 0.000759318, 0.000742815, 0.000726312, 0.000709809, 0.000693305, 0.000676802, 0.000660299, 0.000643795, 0.000627292, 0.000610789, 0.000594286, 0.000577782, 0.000561279, 0.000544776, 0.000528272, 0.000511769, 0.000495266, 0.000478762, 0.000462259, 0.000445756, 0.000429253, 0.000412749, 0.000396246, 0.000379743, 0.000363239, 0.000346736, 0.000330233, 0.000313729, 0.000297226, 0.000280723, 0.00026422, 0.000247716, 0.000231213, 0.00021471, 0.000198206, 0.000181703, 0.0001652, 0.000148696, 0.000132193, 0.00011569, 9.91865e-05, 8.26832e-05, 6.61799e-05, 4.96766e-05, 3.31733e-05], 'lr/pg1': [0.000542744, 0.00108754, 0.00162133, 0.00161749, 0.00160099, 0.00158448, 0.00156798, 0.00155148, 0.00153497, 0.00151847, 0.00150197, 0.00148546, 0.00146896, 0.00145246, 0.00143595, 0.00141945, 0.00140295, 0.00138644, 0.00136994, 0.00135344, 0.00133693, 0.00132043, 0.00130393, 0.00128742, 0.00127092, 0.00125442, 0.00123791, 0.00122141, 0.00120491, 0.0011884, 0.0011719, 0.0011554, 0.00113889, 0.00112239, 0.00110589, 0.00108938, 0.00107288, 0.00105638, 0.00103987, 0.00102337, 0.00100687, 0.000990365, 0.000973861, 0.000957358, 0.000940855, 0.000924352, 0.000907848, 0.000891345, 0.000874842, 0.000858338, 0.000841835, 0.000825332, 0.000808828, 0.000792325, 0.000775822, 0.000759318, 0.000742815, 0.000726312, 0.000709809, 0.000693305, 0.000676802, 0.000660299, 0.000643795, 0.000627292, 0.000610789, 0.000594286, 0.000577782, 0.000561279, 0.000544776, 0.000528272, 0.000511769, 0.000495266, 0.000478762, 0.000462259, 0.000445756, 0.000429253, 0.000412749, 0.000396246, 0.000379743, 0.000363239, 0.000346736, 0.000330233, 0.000313729, 0.000297226, 0.000280723, 0.00026422, 0.000247716, 0.000231213, 0.00021471, 0.000198206, 0.000181703, 0.0001652, 0.000148696, 0.000132193, 0.00011569, 9.91865e-05, 8.26832e-05, 6.61799e-05, 4.96766e-05, 3.31733e-05], 'lr/pg2': [0.000542744, 0.00108754, 0.00162133, 0.00161749, 0.00160099, 0.00158448, 0.00156798, 0.00155148, 0.00153497, 0.00151847, 0.00150197, 0.00148546, 0.00146896, 0.00145246, 0.00143595, 0.00141945, 0.00140295, 0.00138644, 0.00136994, 0.00135344, 0.00133693, 0.00132043, 0.00130393, 0.00128742, 0.00127092, 0.00125442, 0.00123791, 0.00122141, 0.00120491, 0.0011884, 0.0011719, 0.0011554, 0.00113889, 0.00112239, 0.00110589, 0.00108938, 0.00107288, 0.00105638, 0.00103987, 0.00102337, 0.00100687, 0.000990365, 0.000973861, 0.000957358, 0.000940855, 0.000924352, 0.000907848, 0.000891345, 0.000874842, 0.000858338, 0.000841835, 0.000825332, 0.000808828, 0.000792325, 0.000775822, 0.000759318, 0.000742815, 0.000726312, 0.000709809, 0.000693305, 0.000676802, 0.000660299, 0.000643795, 0.000627292, 0.000610789, 0.000594286, 0.000577782, 0.000561279, 0.000544776, 0.000528272, 0.000511769, 0.000495266, 0.000478762, 0.000462259, 0.000445756, 0.000429253, 0.000412749, 0.000396246, 0.000379743, 0.000363239, 0.000346736, 0.000330233, 0.000313729, 0.000297226, 0.000280723, 0.00026422, 0.000247716, 0.000231213, 0.00021471, 0.000198206, 0.000181703, 0.0001652, 0.000148696, 0.000132193, 0.00011569, 9.91865e-05, 8.26832e-05, 6.61799e-05, 4.96766e-05, 3.31733e-05]}, 'git': {'root': '/home/mauriciobenjamin700/projects/companies/iagro/famachapp-models', 'branch': 'main', 'commit': 'a90ef558faeafae8bb9cf869334e9e9e0ff5a20f', 'origin': 'git@github.com:mauriciobenjamin700/famachapp-models.git'}}
(.venv) mauriciobenjamin700@PC:~/projects/courses/pytorch-learning$ 
```

## 🧠 VISÃO GERAL DO MODELO

O modelo treinado é um **YOLO11 ClassificationModel**, com:

* **Tarefa**: classificação
* **Classes**: `2`
* **Entrada**: imagens RGB (`3 canais`)
* **Arquitetura**: CNN profunda com blocos compostos + atenção
* **Flatten explícito?** ❌ não
* **Pooling global?** ✅ sim

Pipeline conceitual:

```bash
(batch, 3, H, W)
↓
Backbone convolucional profundo
↓
Extração semântica + atenção
↓
Global Average Pooling
↓
Linear
↓
(batch, 2)
```

---

### 🧱 BLOCOS REAIS DO MODELO

#### 🔹 BLOCO 0

```bash
Conv2d(3 → 16, stride=2) + BatchNorm + SiLU
```

Shape

```bash
(batch, 3, H, W)
→
(batch, 16, H/2, W/2)
```

Responsabilidade

* Primeira extração de bordas e cores
* Reduz resolução cedo (stride=2)
* BatchNorm estabiliza
* SiLU melhora fluxo de gradiente

📌 **Equivale a**:

```python
Conv → ReLU → Pool
```

Só que fundido e mais eficiente.

---

#### 🔹 BLOCO 1

```text
Conv2d(16 → 32, stride=2) + BN + SiLU
```

Shape

```bash
(batch, 16, H/2, W/2)
→
(batch, 32, H/4, W/4)
```

Responsabilidade

* Extrair padrões mais complexos
* Aumentar canais
* Diminuir espaço

---

#### 🔹 BLOCO 2 — `C3k2`

Esse é o **primeiro bloco composto**.

##### O que é `C3k2`?

É um bloco do tipo:

* Split de canais
* Bottlenecks internos
* Concatenação final

📌 Conceito:

> “Aprender sem perder informação”

Shape

```bash
(batch, 32, H/4, W/4)
→
(batch, 64, H/4, W/4)
```

Responsabilidade

* Aprender relações locais
* Manter eficiência
* Introduzir caminhos residuais

---

#### 🔹 BLOCO 3

```text
Conv2d(64 → 64, stride=2)
```

Shape

```bash
(batch, 64, H/4, W/4)
→
(batch, 64, H/8, W/8)
```

📌 Downsampling explícito

---

#### 🔹 BLOCO 4 — `C3k2`

```bash
64 → 128 canais
```

Shape

```bash
(batch, 64, H/8, W/8)
→
(batch, 128, H/8, W/8)
```

---

#### 🔹 BLOCO 5

```text
Conv2d(128 → 128, stride=2)
```

Shape

```bash
(batch, 128, H/8, W/8)
→
(batch, 128, H/16, W/16)
```

---

#### 🔹 BLOCO 6 — `C3k2` + `C3k`

Mais profundo, mais semântico.

### Shape

```bash
(batch, 128, H/16, W/16)
→
(batch, 128, H/16, W/16)
```

📌 Aqui o modelo **para de reduzir espaço** e foca em **significado**.

---

#### 🔹 BLOCO 7

```bash
Conv2d(128 → 256, stride=2)
```

Shape

```bash
(batch, 128, H/16, W/16)
→
(batch, 256, H/32, W/32)
```

---

#### 🔹 BLOCO 8 — `C3k2`

```bash
(batch, 256, H/32, W/32)
→
(batch, 256, H/32, W/32)
```

---

#### 🔹 BLOCO 9 — `C2PSA` (ATENÇÃO 🚨)

Esse é o bloco mais avançado.

### O que acontece aqui?

* Atenção espacial + canal
* O modelo decide **onde olhar**
* Similar a Transformers, mas convolucional

📌 Responsabilidade:

> Focar nas regiões relevantes da imagem

---

#### 🔹 BLOCO 10 — `Classify` (HEAD)

```bash
Conv 1×1: 256 → 1280
↓
AdaptiveAvgPool2d(1)
↓
Dropout
↓
Linear(1280 → 2)
```

Shape detalhado

```bash
(batch, 256, H/32, W/32)
→ Conv1x1
(batch, 1280, H/32, W/32)
→ GAP
(batch, 1280, 1, 1)
→ Flatten implícito
(batch, 1280)
→ Linear
(batch, 2)
```

📌 **Aqui acontece a classificação final**

---

## 🧠 COMPARAÇÃO COM A CNN “NA MÃO”

| CNN simples        | Seu modelo  |
| ------------------ | ----------- |
| Conv               | Conv        |
| Pool               | Stride      |
| ReLU               | SiLU        |
| Flatten            | GAP         |
| Linear             | Linear      |
| Sem atenção        | Atenção PSA |
| Pouca profundidade | Profundo    |

👉 **Mesmo conceito, engenharia melhor**

---

## 🔑 FRASE-CHAVE PARA FIXAR

> **Seu modelo não decide por pixels,
> ele decide por significado.**

---

## 🎯 CONCLUSÃO

✔ É um **classificador moderno e bem projetado**
✔ Usa **CNN profunda + atenção**
✔ Evita flatten cedo
✔ Muito mais robusto que uma CNN didática
✔ Totalmente alinhado com produção

---
