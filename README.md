# **SDSH: Self-Distilled Self-Supervised Hashing for Remote Sensing Image Retrieval**

Official PyTorch implementation of
**“Self-Distilled Self-Supervised Hashing for Remote Sensing Image Retrieval" (IEEE GRSL 2025)**

![framework](./framework.png)
------

## Environment Setup

This project is built upon Meta AI's **DINOv2** framework.

Please follow the official DINOv2 repository for environment setup instructions:

👉 https://github.com/facebookresearch/dinov2

We recommend using the same dependency versions as specified in the official DINOv2 repository to ensure reproducibility.

## Dataset Preparation

We conduct experiments on the following remote sensing scene classification datasets:

- **AID**
   AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification
   👉 https://captain-whu.github.io/AID/
- **UC Merced Land Use Dataset** (UCMD)
   👉 https://huggingface.co/datasets/blanchon/UC_Merced

Please download the datasets manually and organize them according to your local directory structure.

## Training and Evaluation

### Step 1: Download Pretrained DINOv2 Model

Download the pretrained DINOv2 model:

```
dinov2_vits14_pretrain.pth
```

👉 https://github.com/facebookresearch/dinov2

Place the downloaded checkpoint at your preferred location.

### Step 2: Configure the Training File

Open:

```
dinov2/configs/ssl_default_config.yaml
```

Modify the following fields:

- `student.pretrained_weights`:
   Set this to the path of the downloaded `dinov2_vits14_pretrain.pth`.
- `hash.hash_bit`:
   Set the hash code length. Supported options:
   `16`, `32`, `48`, `64`
- `train.data_path`:
   Set this to the root directory of AID or UCMD.

#### Dataset-specific settings

For **AID**:

- `train.train_ratio = 0.5`
- `OFFICIAL_EPOCH_LENGTH = 80`
- `evaluation.eval_period_iterations = 80`

For **UCMD**:

- `train.train_ratio = 0.8`
- `OFFICIAL_EPOCH_LENGTH = 30`
- `evaluation.eval_period_iterations = 30`



### Step 3: Run Training and Evaluation

Execute the following command:

```
python dinov2/train/train.py \
    --config-file <path_to_your_config> \
    --output-dir <path_to_save_logs_and_checkpoints>
```

The output directory will contain training logs and model checkpoints.

------

## 📚 **Citation**

If you find this work useful, please cite:

```bibtex
@ARTICLE{11303888,
  author={Qiao, Shishi and Yuan, Shuai and Fu, Hang and Yan, Mai and Zheng, Haiyong},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Self-Distilled Self-Supervised Hashing for Remote Sensing Image Retrieval}, 
  year={2026},
  volume={23},
  number={},
  pages={1-5},
  keywords={Image retrieval;Remote sensing;Training;Binary codes;Visualization;Sensors;Annotations;Semantics;Manuals;Feature extraction;Attention;remote sensing (RS) image retrieval;self-distilled self-supervised hashing (SDSH)},
  doi={10.1109/LGRS.2025.3646078}}

```

------

## 📜 **License**

This project is released under the MIT License.