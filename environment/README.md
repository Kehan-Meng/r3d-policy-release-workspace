# Reproducible Environment

Validated core stack:

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- CUDA toolkit 12.1
- GCC/G++ 11
- NVIDIA driver new enough for CUDA 12.1

Create the Conda environment, then compile the two CUDA extensions against the
installed Torch ABI:

```bash
conda env create -f environment/environment.yml
conda activate r3d-release
bash environment/install.sh
python download_pretrained.py
```

The order matters. `pytorch3d_simplified` and `pointnet2_ops` cannot be safely
represented as ordinary cross-platform wheels: their binaries depend on the
Torch, CUDA, compiler, and GPU architecture available on the target machine.
`environment/install.sh` builds them from the released sources and never copies
the original server's `.so` files.

Benchmark simulators are separate because their dependency stacks and assets
are large and mutually sensitive:

```bash
bash environment/install_benchmarks.sh metaworld
bash environment/install_benchmarks.sh adroit
bash environment/install_benchmarks.sh maniskill2
bash environment/install_benchmarks.sh robotwin2
```

Then verify only the benchmark you plan to run:

```bash
python environment/verify_environment.py --benchmark metaworld
```
