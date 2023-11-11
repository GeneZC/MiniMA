
pip install crc32c
pip install sklearn
pip install sentencepiece
pip install transformers==4.34.0
pip install accelerate
pip install bitsandbytes
pip install triton==2.1.0
pip install packaging
pip install ninja
pip install flash-attn==2.0.5 --no-build-isolation
pip uninstall -y transformer_engine
pip install deepspeed==0.10.0
# cd modules/ops/layernorm; pip install .