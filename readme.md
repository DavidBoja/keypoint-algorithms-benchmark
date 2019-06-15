## Detector-descriptor benchmark

This is a benchmark for evaluation of state-of-the-art detector and descriptor algorithms. The details about the benchmark will soon be published as a paper.

### HPSequences dataset

![hpsequences](assets/sequences.png)

Download dataset: [HPSequences dataset](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) [1.3GB]

After download, the folder `../data/hpatches-release` contains all the
patches from the 116 sequences. The sequence folders are named with
the following convention

* `i_X`: patches extracted from image sequences with illumination changes
* `v_X`: patches extracted from image sequences with viewpoint changes

### Results

To be published...

### Remarks

This benchmark is based on the HPatches evaluation tasks [[1]](#refs) and HPSequences dataset published along with it ([HPatches dataset repository](https://github.com/hpatches/hpatches-dataset)). Thanks to the authors for providing the dataset and the evaluation details.

### References
<a name="refs"></a>

[1] *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*, Vassileios Balntas*, Karel Lenc*, Andrea Vedaldi and Krystian Mikolajczyk, CVPR 2017.
*Authors contributed equally.
