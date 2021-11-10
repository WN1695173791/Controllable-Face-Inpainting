# Facial Image Inpainting with Semantic Control
In this repo, we provide a model for the controllable facial image inpainting task. This model enables users to intuitively edit their images by using parametric 3D faces. 

The technology report is comming soon.

* **Image Inpainting results**

  <p align='center'>  
    <img src='https://user-images.githubusercontent.com/30292465/141147333-ce25efab-2434-4674-b43d-398758fa0834.png' width='500'/>
  </p>

* **Fine-grained Control**

  <p align='center'>  
    <img src='https://user-images.githubusercontent.com/30292465/141147069-d119b408-151c-4d28-a7d6-7ea01cc02daf.png' width='700'/>
  </p>




## Quick Start

### Installation

* Clone the repository and set up a conda environment with all dependencies as follows

```bash
git clone https://github.com/RenYurui/Controllable-Face-Inpainting.git --recursive
cd Controllable-Face-Inpainting

# 1. Create a conda virtual environment.
conda create -n cfi python=3.6
source activate cfi
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2

# 2. install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

# 3. Install other dependencies
pip install -r requirements.txt
```

### Download Prerequisite Models 

* Follow [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch#prepare-prerequisite-models) to prepare `./BFM` folder. Download `01_MorphableModel.mat` and Expression Basis `Exp_Pca.bin`. Put the obtained files into the `./Deep3DFaceRecon_pytorch/BFM` floder. Then link the folder to the root path.

```bash
ln -s /PATH_TO_REPO_ROOT/Deep3DFaceRecon_pytorch/BFM /PATH_TO_REPO_ROOT
```

* Clone the Arcface repo


```bash
cd third_part
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch/ ./
```

The [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) is used to extract identity features for loss computation. Download the pre-trained model from Arcface using this [link](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3). By default, the resnet50 backbone ([ms1mv3_arcface_r50_fp16](https://onedrive.live.com/?authkey=!AFZjr283nwZHqbA&id=4A83B6B633B029CC!5583&cid=4A83B6B633B029CC)) is used. Put the obtained weights into `./third_part/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth`			

* Download the pretrained weights of our model from [Google Driven](https://drive.google.com/drive/folders/1fyVI81I5gP4is4zN2kvM3WiMRdPXguVf?usp=sharing). Save the obtained files into folder `./result`.



###  Inference 

We provide some example images. Please run the following code for inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 1234 demo.py \
--config ./config/facial_image_renderer_ffhq.yaml \
--name facial_image_renderer_ffhq \
--output_dir ./visi_result \
--input_dir ./examples/inputs \
--mask_dir ./examples/masks
```



## Train the model from scratch

### Dataset Preparation

* Download dataset. We use [Celeba-HQ](https://github.com/tkarras/progressive_growing_of_gans) and [FFHQ](https://github.com/NVlabs/ffhq-dataset) for training and inference. Please download the datasets (image format) and put them under .`/dataset` folder.
* Obtain 3D faces by using [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch).  Follow the Deep3DFaceRecon repo to download the trained weights. And save it as: `./Deep3DFaceRecon_pytorch/checkpoints/face_recon/epoch_20.pth`


```bash
# 1. Extract keypoints from the face images for cropping.
cd scripts
# extracted keypoints from celeba
python extract_kp.py \
--data_root PATH_TO_CELEBA_ROOT \
--output_dir PATH_TO_KEYPOINTS \
--dataset celeba \
--device_ids 0,1 \
--workers 6

# 2. Extract 3DMM coefficients from the face images.
cd .. #repo root
# we provide some scripts for easy of use. However, one can use the original repo to extract the coefficients.
cp scripts/inference_options.py ./Deep3DFaceRecon_pytorch/options
cp scripts/face_recon.py ./Deep3DFaceRecon_pytorch
cp scripts/facerecon_inference_model.py ./Deep3DFaceRecon_pytorch/models
cp scripts/pytorch_3d.py ./Deep3DFaceRecon_pytorch/util
ln -s /PATH_TO_REPO_ROOT/third_part/arcface_torch /PATH_TO_REPO_ROOT/Deep3DFaceRecon_pytorch/models

cd Deep3DFaceRecon_pytorch

python face_recon.py \
--input_dir PATH_TO_CELEBA_ROOT \
--keypoint_dir PATH_TO_KEYPOINTS \
--output_dir PATH_TO_3DMM_COEFFICIENT \
--inference_batch_size 100 \
--name=face_recon \
--dataset_name celeba \
--epoch=20 \
--model facerecon_inference

# 3. Save images and the coefficients into a lmdb file.
cd .. #repo root
python prepare_data.py \
--root PATH_TO_CELEBA_ROOT \
--coeff_file PATH_TO_3DMM_COEFFICIENT \
--dataset celeba \
--out PATH_TO_CELEBA_LMDB_ROOT
```



### Train The Model 

```bash
# we first train the semantic_descriptor_recommender
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1234 train.py \
--config ./config/semantic_descriptor_recommender_celeba.yaml \
--name semantic_descriptor_recommender_celeba

# Then, we trian the facial_image_renderer for image inpainting
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 1234 train.py \
--config ./config/facial_image_renderer_celeba.yaml \
--name facial_image_renderer_celeba
```
