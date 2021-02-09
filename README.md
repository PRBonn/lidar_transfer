# LiDAR-Transfer

Transfer SemanticKITTI labeles into other dataset/sensor formats.

![Motivation](motivation.svg)

## Content
- Convert datasets ([NUSCENES](auxiliary/convert/nuscenes2kitty.py), [FORD](auxiliary/convert/ford2kitty.py), [NCLT](auxiliary/convert/nclt2kitty.py)) to KITTI format
- Minimal dataset [minimal.zip](minimal.zip)
- Visualize with [visualizer.py](visualizer.py)
- Transfer datasets [lidar_deform.py](lidar_deform.py)

## Usage
<details>
<summary>Install Dependencies</summary>

```
pip install pyaml pyqt5 scikit-image scipy torchvision

pip install pycuda

pip install vispy
```

Or use local installation to apply antialias patch

```
git clone https://github.com/vispy/vispy.git
cd vispy
git apply ../lidar_transfer/vispy_antialias.patch
pip install -e .
```

</details>

<details>
<summary>Get started</summary>

1. Unzip `minimal.zip`
2. Run 
    ```
    python lidar_deform.py -d minimal
    ```
3. Run with target sensor
    ```
    python lidar_deform.py -d minimal -t minimal/target.yaml
    ```
4. Change parameter in `config/lidar_transfer.yaml`

</details>

## Credits
Developed by Ferdinand Langer, 2019.
This tool uses the following open source software:
- Fast BVH by Brandon Pelfrey https://github.com/brandonpelfrey/Fast-BVH
- TSDF Fusion by Andy Zeng https://github.com/andyzeng/tsdf-fusion-python
- Visualization http://vispy.org/

## License

Copyright 2020, Ferdinand Langer, Cyrill Stachniss. University of Bonn.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation

When you use our code in any academic work, please cite the original [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/langer2020iros.pdf).

```
@inproceedings{langer2020iros,
    author = {F. Langer and A. Milioto and A. Haag and J. Behley and C. Stachniss},
    title = {{Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks}},
    booktitle = {Proc.~of the IEEE/RSJ Intl. Conf. on Intelligent Robots and System (IROS)},
    year = {2020},
    url = {http://www.ipb.uni-bonn.de/pdfs/langer2020iros.pdf},
    videourl = {https://youtu.be/6FNGF4hKBD0},
}
```