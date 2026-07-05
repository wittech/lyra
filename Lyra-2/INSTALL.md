# Installation

Tested on Ubuntu 22.04 with CUDA 12.8 and NVIDIA H100 GPUs. Other Linux distributions and CUDA 12.4+ should also work but are not officially verified. Due to the complexity of the dependency stack, version conflicts may arise on different system configurations — if so, please open an issue.

```bash
cd /data/projects/lyra/Lyra-2/

# 0. Clone repository
git clone --recursive https://github.com/wittech/lyra.git
cd Lyra-2

# 1. Create conda environment
conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
conda activate lyra2
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y

# 2. Install CUDA toolkit inside the conda environment
# conda install cuda -c nvidia/label/cuda-12.8.0 -y
# export CUDA_HOME=$CONDA_PREFIX
# 1. 退出当前环境
conda deactivate

# 2. 删除无效PATH配置
conda env config vars unset -p /data/opt/miniconda3/envs/lyra2 PATH

# 3. 重新设置允许的CUDA相关变量
conda env config vars set -p /data/opt/miniconda3/envs/lyra2 \
CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# 4. 创建激活钩子自动补PATH
mkdir -p /data/opt/miniconda3/envs/lyra2/etc/conda/activate.d
echo '#!/bin/bash
export PATH="/usr/local/cuda-12.8/bin:$PATH"' > /data/opt/miniconda3/envs/lyra2/etc/conda/activate.d/cuda_path.sh
chmod +x /data/opt/miniconda3/envs/lyra2/etc/conda/activate.d/cuda_path.sh

# 5. 重新激活验证
conda activate lyra2
nvcc -V
echo $CUDA_HOME

# 3. Install PyTorch
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

# 4. Set build environment variables
# 头文件路径：仅系统cuda12.8（自带cudnn、nccl头文件）
export CPATH="$CUDA_HOME/include:$CPATH"

# 库路径：
# 1. CUDA12.8 lib64（cudnn+nccl+cuda库）置顶
# 2. conda自身python/torch库保留
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# 5. Install Python dependencies
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
# pip install --no-build-isolation "transformer_engine[pytorch]"
GIT_CONFIG_PARAMETERS='url."https://ghfast.top//https://github.com/".insteadOf=https://github.com/' pip install --no-build-isolation "transformer-engine[core_cu12,pytorch]"

# # Symlink cuda_runtime as cudart for transformer_engine compatibility
# SITE=$CONDA_PREFIX/lib/python3.10/site-packages
# ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# 6. Install Flash Attention
export FLASH_ATTN_CUDA_ARCHS="90"
export TORCH_CUDA_ARCH_LIST="9.0"
export NVCC_THREADS=16
export MAX_JOBS=20
GIT_CONFIG_PARAMETERS='url."https://ghfast.top//https://github.com/".insteadOf=https://github.com/' pip install --no-build-isolation --no-binary :all: flash-attn==2.8.3

# 7. Build vendored CUDA extensions
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install -r /data/projects/lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3/requirements.txt
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

# 8. 设置环境变量
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$SITE/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 9. 设置模型
(lyra2) root@ubuntu:/data/projects/lyra/Lyra-2# ln -s /data/models/Lyra-2.0/checkpoints/ .


```
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('all imports OK')
"
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```

