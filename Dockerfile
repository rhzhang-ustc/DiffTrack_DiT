# FROM pytorch/pytorch:pytorch-1.9.0-cuda11.3-cudnn8-devel

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

#ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get update && apt-get install -y libibverbs1

ENV TZ=US/Pacific

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezon

RUN apt-get update && apt-get install -y libibverbs1 ibverbs-providers

 

# run slowfast dependency

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

RUN apt-get update && apt-get install -y libsm6 libxext6 git-all

ENV MKL_SERVICE_FORCE_INTEL=1

 

 

# RUN pip install fvcore Pillow simplejson psutil opencv-python tensorboard && \

#     pip install av && \

#     conda install -c conda-forge ffmpeg && \

#     conda install -c iopath iopath && \

#     conda install -c conda-forge moviepy && \

#     #pip install 'git+https://github.com/facebookresearch/fairscale' && \

#     git clone https://github.com/facebookresearch/detectron2 detectron2_repo && pip install -e detectron2_repo && \

#     pip install einops decord

RUN pip install fvcore wandb tqdm numpy && \
    pip install scipy shapely torchvision &&\
    pip install timm matplotlib pillow &&\
    pip install pyyaml configparser pyzmq ftfy &&\
    pip install opencv-python

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
 
ENV SINGULARITY_IMAGE_FRAMEWORK="PYTORCH"

ENV SINGULARITY_IMAGE_ACCELERATOR="NVIDIA"

 

ENV SINGULARITY_IMAGE_ACCELERATOR_SKU="V100"

 

USER root

WORKDIR /home/aiscuser

## usage
## docker build -t tangchuanxin/slowfast:deepspeed .

## docker login

## docker push tangchuanxin/slowfast:deepspeed