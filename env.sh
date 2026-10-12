# Source before any work on worf:  source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
export PROJ=/scratch/ybicakci/projects/elevate_amsterdam
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch/ybicakci/hf_home                # models shared with earlier work
export UV_LINK_MODE=hardlink   # uv cache and envs share /scratch, so no duplicate files
export UV_CACHE_DIR=/scratch/ybicakci/cache/uv          # keep the uv cache off the root disk
export VLLM_CACHE_ROOT=/scratch/ybicakci/cache/vllm
export TRITON_CACHE_DIR=/scratch/ybicakci/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/ybicakci/cache/inductor
export FLASHINFER_WORKSPACE_BASE=/scratch/ybicakci/cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0                    # FlashInfer's JIT sampler needs ninja
# Three environments because their transformers pins conflict.
export PY_VLM=$PROJ/envs/vlm/bin/python                 # vLLM 0.31 (transformers < 5.18)
export PY_EMBED=$PROJ/envs/embed/bin/python             # EmbeddingGemma 2 (transformers >= 5.18)
export PY_IMAGE=$PROJ/envs/image/bin/python             # Qwen-Image-2.1 (diffusers from git)
export PY_PERCEPT=$PROJ/envs/percept/bin/python         # svi_percept: open_clip ViT-H-14-378 (step 2.4b)
cd $PROJ
