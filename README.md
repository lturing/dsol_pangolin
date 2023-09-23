DSOL是一种适用于嵌入式芯片上的稀疏直接法的VO。相比官方的代码(ros)，我们用pangolin可视化。

# run
```
# install dependencies

## install glog
git clone --depth 1 --branch v0.6.0 https://github.com/google/glog.git
cd glog
mkdir build && cd build 
cmake -DBUILD_SHARED_LIBS=TRUE ..
make -j6
sudo make install

## install fmt
git clone --depth 1 --branch 8.1.0 https://github.com/fmtlib/fmt.git
cd fmt
mkdir build && cd build 
cmake -DBUILD_SHARED_LIBS=TRUE -DCMAKE_CXX_STANDARD=17 -DFMT_TEST=False ..
make -j6
sudo make install

## Install Abseil
git clone --depth 1 --branch 20220623.0 https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
mkdir build && cd build
cmake -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=TRUE ..
make -j6
sudo make install

## Install Sophus
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build && cd build
git checkout 785fef3
cmake -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF -DCMAKE_CXX_STANDARD=17 ..
make -j6
sudo make install


## Install Google benchmark
git clone --depth 1 --branch v1.6.2 https://github.com/google/benchmark.git
cd benchmark
mkdir build && cd build 
cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBENCHMARK_ENABLE_GTEST_TESTS=OFF ..
make -j12
sudo make install 


## clone this codes
git clone 
cd 
mkdir build 
cmake .. && make -j12

## run kitti00
cd ..
./example/dsol_kitti

```

# todo 
- add loop close 
- add imu

----------------

# 🛢️ DSOL: Direct Sparse Odometry Lite

## Reference

Chao Qu, Shreyas S. Shivakumar, Ian D. Miller, Camillo J. Taylor

https://arxiv.org/abs/2203.08182

https://youtu.be/yunBYUACUdg

## Datasets

VKITTI2 https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

KITTI Odom 

TartanAir https://theairlab.org/tartanair-dataset/

Sample realsense data at

https://www.dropbox.com/s/bidng4gteeh8sx3/20220307_172336.bag?dl=0

https://www.dropbox.com/s/e8aefoji684dp3r/20220307_171655.bag?dl=0

Calib for realsense data is 

```
393.4910888671875 393.4910888671875 318.6263122558594 240.12942504882812 0.095150406
```
Put this in `calib.txt` and put it in the same folder of the realsense dataset generated by the python file.

## Build

This is a ros package, just put in a catkin workspace and build the workspace.

## Run
Open rviz using the config in `launch/dsol.rviz`

```
roslaunch dsol dsol_data.launch
```

See launch files for more details on different datasets.

See config folder for details on configs.

To run multithread and show timing every 5 frames do
```
roslaunch dsol dsol_data.launch tbb:=1 log:=5
```

## Dependencies and Install instructions

See CMakeLists.txt for dependencies. You may also check our [Github Action build
file](https://github.com/versatran01/dsol/blob/main/.github/workflows/build.yaml) for instructions on how to build DSOL in Ubuntu 20.04 with ROS Noetic.

## Disclaimer

For reproducing the results in the paper, place use the `iros22` branch.

This is the open-source version, advanced features are not included.

## Related

See here for a fast lidar odometry

https://github.com/versatran01/rofl-beta

https://github.com/versatran01/llol
