#!/bin/bash
# Step 0.2: three environments (their transformers pins conflict) + smoke tests.
set -e
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
for e in vlm embed image; do [ -d envs/$e ] || uv venv --python 3.12 envs/$e; done
echo "== vlm";   uv pip install --python envs/vlm/bin/python -e "GeoAI-VLM[vlm,segment,network]" mapbox-vector-tile num2words pytest 2>&1 | tail -2
echo "== embed"; uv pip install --python envs/embed/bin/python -e "GeoAI-VLM[sentence-transformers,segment,network]" "transformers>=5.19" pytest 2>&1 | tail -2
echo "== image"; uv pip install --python envs/image/bin/python "git+https://github.com/huggingface/diffusers.git" "transformers>=5.17" accelerate pillow torch torchvision -e "GeoAI-VLM[segment]" 2>&1 | tail -2
for e in vlm embed image; do envs/$e/bin/python -c "import torch, transformers; print('$e', 'torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count())"; done
envs/vlm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
envs/embed/bin/python -c "import sentence_transformers, torchvision; print('st', sentence_transformers.__version__, 'torchvision', torchvision.__version__)"
envs/image/bin/python -c "import diffusers; from diffusers import QwenImage21Pipeline; print('diffusers', diffusers.__version__, 'QwenImage21Pipeline ok')"
echo SETUP_DONE
