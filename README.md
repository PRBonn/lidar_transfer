# lidar_transfer

Code for Ferdinand Langer master thesis.
Transfer semantic KITTI labeles into other dataset/sensor formats.

## Content
- Convert datasets ([NUSCENES](nuscenes2kitty.py), [FORD](ford2kitty.py), [NCLT](nclt2kitty.py)) to KITTI format
- Visualize with [visualizer.py](visualizer.py)
- Deform datasets

## ToDo
- [x] Config files for different sensors/datasets
    - [x] source config withing dataset/sequence/00/config.yaml
    - [x] target config pass scanner.yaml as argument
- [x] Tool to convert to KITTI structure
    - [x] nuscenes2kitti
    - [x] ford2kitti
    - [x] nclt2kitti
- [x] Visualize
- [ ] Deformed
    - [x] Closest point
    - [ ] Mesh
    - [ ] Categorie mesh
    - [x] use aggregated point cloud


