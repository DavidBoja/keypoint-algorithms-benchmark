## Detector-descriptor benchmark

This is a benchmark for evaluation of state-of-the-art detector and descriptor algorithms. The details about the benchmark will soon be published as a paper.

### HPSequences dataset

Download dataset: [HPSequences dataset](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz) [1.3GB]

The data required for the benchmarks are saved in the `./data` folder,
and are shared between the two implementations.

To download the `HPatches` image dataset, run the provided shell script
with the `hpatches` argument.

``` bash
sh download.sh hpatches
```
To download the pre-computed files of a baseline descriptor `X` on the
`HPatches` dataset, run the provided `download.sh` script with the
`descr X` argument.  

To see a list of all the currently available descriptor file results,
run scipt with only the `descr` argument.

``` bash sh 
sh download.sh descr       # prints all the currently available baseline pre-computed descriptors
sh download.sh descr sift  # downloads the pre-computed descriptors for sift
```

The `HPatches` dataset is saved on `./data/hpatches-release` and the pre-computed descriptor files are saved on `./data/descriptors`.


### Dataset description

After download, the folder `../data/hpatches-release` contains all the
patches from the 116 sequences. The sequence folders are named with
the following convention

* `i_X`: patches extracted from image sequences with illumination changes
* `v_X`: patches extracted from image sequences with viewpoint changes

### Remarks

This benchmark is based on the HPatches evaluation tasks [[1]](#refs) and HPSequences dataset published along with it ([HPatches dataset repository](https://github.com/hpatches/hpatches-dataset)). Thanks to the authors for providing the dataset and the evaluation details.

### References
<a name="refs"></a>

[1] *HPatches: A benchmark and evaluation of handcrafted and learned local descriptors*, Vassileios Balntas*, Karel Lenc*, Andrea Vedaldi and Krystian Mikolajczyk, CVPR 2017.
*Authors contributed equally.

