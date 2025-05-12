## Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement


Yini Li, Nantheera Anantrasirichai

[[Paper]](https://arxiv.org/abs/2503.11175)

---
## Overview
![](.\illustration\Network_pipeline.png)
We proposed Zero-TIG, a zero-shot self-supervised
method for low-light video enhancement. Additionally, an adaptive white
balance is introduced for underwater data, achieving temporal
coherence and color accuracy without paired training data.

---

## Dataset


For BVI-RLV dataset, please refer to the [website](https://ieee-dataport.org/open-access/bvi-lowlight-fully-registered-datasets-low-light-image-and-video-enhancement).

Data Structure:
```
.
└── BVI-RLV dataset
    ├── input
    │   ├── S01
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   ├── S02
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   └── ...
    └── gt
        ├── S01
        │   ├── normal_light_10
        │   └── normal_light_20
        ├── S02
        │   ├── normal_light_10
        │   └── normal_light_20
        └── ...
```

---

## Training
- For the training of BVI-RLV, you can refer to the command below:
```shell
python train.py --lowlight_images_path <path_to_folder> --dataset RLV
```

- For the training of underwater data, you can refer to the command below:
```shell
python train.py --lowlight_images_path <path_to_folder> --dataset underwater
```

- If you want to train your own dataset, please adapt the `dataloader\multi_read_data.py` to your dataset.

---

## Prediction
- For the prediction of BVI-RLV, you can refer to the command below:
```shell
python predict.py --lowlight_images_path <path_to_folder> --model_pretrain <path_to_model> --dataset RLV
```

- For the prediction of underwater data, you can refer to the command below:
```shell
python train.py --lowlight_images_path <path_to_folder> --model_pretrain <path_to_model> --dataset underwater
```

---

## Citation

```bibtex
@misc{li2025zerotigtemporalconsistencyawarezeroshot,
      title={Zero-TIG: Temporal Consistency-Aware Zero-Shot Illumination-Guided Low-light Video Enhancement}, 
      author={Yini Li and Nantheera Anantrasirichai},
      year={2025},
      eprint={2503.11175},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11175}, 
}
```