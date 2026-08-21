# Third-Party Components

`pytorch3d_simplified` contains only the FPS operator used by this project and
is derived from Meta's PyTorch3D under its BSD license. `pointnet2_ops` is
derived from `erikwijmans/Pointnet2_PyTorch` under the Unlicense and contains
the CUDA PointNet2 kernels used by PointSAM. Their original license texts are
kept in each source directory. Both are compiled locally by
`environment/install.sh` so the release does not ship a machine-specific CUDA
binary.

Benchmark repositories and their assets are not copied into Git. Run
`environment/install_benchmarks.sh`; it checks out the exact integration
versions under this directory. RoboTwin2 is pinned to
`74f4e99720b4b296a38af9e95ee31d9e400073af`, whose task files match the R3D
integration manager byte-for-byte.
