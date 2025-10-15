
### Before Running Files
1. Create venv with ```python -m venv venv```
2. Use the venv with ```./venv/Scripts/activate``` if you're on Windows, or ```source ./venv/bin/activate``` on Linux/Mac
3. Install necessary dependencies with ```python -m pip install -r requirements.txt```


### Compressed XOR Operation for GEMM
- File: xor_zstd_transformer.ipynb
- Essentially performs tiled decompression of zstd, multiplies the certain tile, and then compresses it to save storage in memory
- Once complete, this will be applied to the greater picture within xor_zstd_transformer_demo.ipynb

### Intermediary Operation between Matrices:
- File: xor_intermediate_op.ipynb
- Essentially solves for x (a XOR b) using the existing computer calculation formula for a + b.

### Visualization
- File: visualize.ipynb
- As of now, this is a basic visualization of how the delta will be used to store fine tuned models (from base to fine-tuned visual)
- To be built out once the Compressed XOR GEMM is complete

