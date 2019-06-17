# lidar_transfer

Code for Ferdinand Langer master thesis.
Transfer semantic KITTI labeles into other dataset/sensor formats.

## Content
- Convert datasets ([NUSCENES](nuscenes2kitty.py), [FORD](ford2kitty.py), [NCLT](nclt2kitty.py)) to KITTI format
- Visualize with [visualizer.py](visualizer.py)
- Deform datasets

## ToDo
- [ ] Config files for different sensors/datasets
- [x] Tool to convert to KITTI structure
    - [x] nuscenes2kitti
    - [x] ford2kitti
    - [x] nclt2kitti
- [ ] Visualize
- [ ] Deformed
    - [ ] Closest point
    - [ ] Mesh
    - [ ] Categorie mesh
    - [ ] use aggregated point cloud


