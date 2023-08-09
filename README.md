# SwinSUNet

1.Install
Pytorch 1.7.1
Torchvision 0.8.2
Python 3.7.10
einops
numpy
matplotlib
scikit-image
tqdm

2.Dataset

2.1Image storage format
|PATH_TO_DATASET_train/
|----A/
|  |----IMG1.jpg
|  |----IMG2.jpg
|  |----...
|  |----IMGn_train.jpg
|----B/
|  |----IMG1.jpg
|  |----IMG2.jpg
|  |----...
|  |----IMGn_train.jpg
|----label/
|  |----IMG1.jpg
|  |----IMG2.jpg
|  |----...
|  |----IMGn_train.jpg
|----img.txt
|PATH_TO_DATASET_test/
|----A/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_test.jpg
|----B/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_test.jpg
|----label/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_test.jpg
|----img.txt
|PATH_TO_DATASET_val/
|----A/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_val.jpg
|----B/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_val.jpg
|----label/
|  |----IMG1.jpg
|  |----...
|  |----IMGn_val.jpg
|----img.txt

2.2LEVIR-CD dataset
LEVIR-CD can be download from https://justchenhao.github.io/LEVIR/

2.3Img.txt
2.3.1Content of img.txt
IMG1.jpg
IMG2.jpg
IMG3.jpg
...
IMGn.jpg
2.3.2Generation of img.txt
$cd PATH_TO_DATASET_train/A
$ls A > ../img.txt
