## Data-Efficient Multi-Scale Fusion Vision Transformer

This repository includes official implementation and model weights of Data-Efficient Multi-Scale Fusion Vision Transformer.

### 1. Abstract

Vision transformer (ViT) demonstrates significant potential in image classification with massive data, but struggles with small-scale datasets. 
To this end, this paper proposes to address this data ineﬀiciency by introducing multi-scale tokens, which provides the image prior of multiple scales and enables learning scale-invariant features. Our model generates tokens of varying scales from images using different patch sizes, where each token of the larger scale is linked to a set of tokens of other smaller scales based on spatial correspondences. Through a regional cross-scale interaction module, tokens of different scales fuse regionally to enhance the learning of local structures.Additionally, we implement a data augmentation schedule to refine training. Extensive experiments on image classification demonstrate our approach surpasses DeiT and other multi-scale transformer methods on small-scale datasets.
+ **Multi-Scale Tokenization**
<div align="center">
  <img src="./images/dems_plot1.jpg" width="600px" />
</div>

+ **Regional Cross-Scale Interaction**
<div align="center">
  <img src="./images/dems_plot2.jpg" width="600px" />
</div>

### 2. Requirements

To install requirements:

```setup
conda create -n dems python=3.8
pip install -r requirements.txt
```

### 3. Datasets

The root paths of data are set to `/path/to/dataset`. Please set the root paths accordingly.
`CIFAR10`, `CIFAR100`, `FashionMNIST`, `EMNIST` datasets provided by `torchvision`. 
Download and extract Caltech101 train and val images from https://www.vision.caltech.edu/datasets/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively.

### 4. Training

Set hyperparameters and GPU IDs in `./config/pretrain/dems_small_pretrain.py`.
Run the following command to train DEMS-ViT-S on CIFAR100 for 800 epochs, with random initialization on a single node with multiple gpus:

```shell script

python main_pretrain --model dems_small --batch_size 256 --epochs 800 --dataset CIFAR100 --data_path /path/to/CIFAR100
```

### 5. Fine-tuning

Set hyperparameters and GPU IDs in `./config/pretrain/dems_small_finetune.py`.
Run the following command to finetune DEMS-ViT-S on CIFAR100 for 100 epochs:

```shell script

python main_finetune --model dems_small --batch_size 256 --epochs 100 --dataset CIFAR100 --data_path /path/to/CIFAR100 --pretrained_weight /path/pretrained
```

### 6. Main Results and Model Weights

#### 6.1 Pretrained weights

We provide models trained on CIFAR, EMNIST, FASHIONNIST, and CALTECH101 [here](https://drive.google.com/drive/folders/1hOHYIBBnICmtynDxgEQ1j_GAHV7xSrAq?usp=sharing).
Particularly, we train on CALTECH101 with the input size of 256x256 and patch size of 16.

| Name | #FLOPs | #Params | Dataset | Acc@1 | URL |
| --- | --- | --- | --- | --- | --- |
| DEMS-ViT-Ti | 1.6 | 5.6M |CIFAR10<br>CIFAR100<br>FASHIONMNIST<br>EMNIST<br>CALTECH101 | 96.03<br>80.60<br>95.59<br>99.56<br>86.56 | [model](https://drive.google.com/file/d/1fjvv1tjIHGhRNfzimbC5BHGclDYNGQcf/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1i9PYlxYNH9z0jGDiG1uEWOxxpd5yR7k9/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1QNVIDtKz5xOVTzgq1YAUwXnbAiBWO8AP/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1UzHl93KSsQ3BAXstYIZ2BUKXc1L_H4iI/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1PSJiop0uipScOfPoxPagjcx5g89biqWa/view?usp=sharing) |
| DEMS-ViT-S | 5.8 | 22.3M |  CIFAR10<br>CIFAR100<br>FASHIONMNIST<br>EMNIST<br>CALTECH101 | 96.20<br>83.30<br>95.99<br>99.58<br>86.88 | [model](https://drive.google.com/file/d/1A3BT0xkj_7hL_FjGJRK23uPOya1TZta4/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1vqWTQleKpk9AMxozk6U_E9YV8UYZLWrI/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1cgM3ubdQg-yGZRj9Zc6odV0wDC57sPTj/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1UzHl93KSsQ3BAXstYIZ2BUKXc1L_H4iI/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1HHPUE79hKsL2fe9E-VzxtImheIKPLqKw/view?usp=sharing) |

#### 6.2 Fine-tuned weights

We provide fine-tuned models on CIFAR, which can be found [here](https://drive.google.com/drive/folders/14klxjyBhq-P_8QVB5oqEFOGsn6wYydnt?usp=sharing).

| Name | Dataset | Acc@1 | URL |
| --- | --- | --- | --- |
| DEMS-ViT-Ti | CIFAR10<br>CIFAR100 | 96.74<br>83.50 | [model](https://drive.google.com/file/d/1mvVLZamCz9NuSssrxRRtVIii19Uj0P8n/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1Qtr1E04bveXopxVlsxDCFAJ3hXc8odIl/view?usp=sharing)|
| DEMS-ViT-S | CIFAR10<br>CIFAR100 | 97.76<br>85.16 | [model](https://drive.google.com/file/d/11jmSisi2RXP9DORI5hdZRDtM8vBZDTf4/view?usp=sharing)<br>[model](https://drive.google.com/file/d/1HsSvST0VI6K_j946VtOj81lOxnOjSmn1/view?usp=sharing)|

### 7. License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](./LICENSE) for details.
