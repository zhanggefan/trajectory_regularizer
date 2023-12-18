# MOTREG: MOTion model for 3D box trajectory REGularization

MOTREG：用于3D框轨迹整定的运动学模型

## 简介

该库实现了基于CTRV的运动学模型，构造最小二乘问题，对运动目标进行运动轨迹拟合。模型的数学原理并不复杂，都是简单的李代数求导，可惜没有能读的文档了   
该库可以实现：

- 使用运动学模型拟合3D框轨迹，对各3D框进行微调整定，使得其满足刚体运动约束
- 锁死轨迹中所有3D框，并使用运动学模型拟合轨迹，通过拟合残差发现不符合刚体运动规律的3D框
- 锁死轨迹中部分3D框，并使用运动学模型优化其余未锁死的3D框，实现交互式轨迹整定
- 给定轨迹的首、尾两个3D框，根据运动学模型插补中间的3D框
- 使用运动学模型拟合轨迹，得到轨迹中每个3D框的速度估计
- CPU端单次拟合优化耗时不长于100ms，可做到准实时交互

## 编译

该库依赖CMake进行编译。第三方依赖库包含：

* suitesparse —— 稀疏矩阵求解库，用于提升g2o的求解速度，需要通过vcpkg安装
* eigen3 —— 矩阵运算库，需要通过vcpkg安装
* fmt —— cpp字符串格式化库，需要通过vcpkg安装
* sophus —— 李代数运算库，已经是本仓库的git submodule
* g2o —— 图优化库，已经是本仓库的git submodule

### Windows下的编译过程

#### 步骤一：安装vcpkg（如果已有vcpkg则可跳过此步）

```
> git clone https://github.com/microsoft/vcpkg
> cd vcpkg
> .\bootstrap-vcpkg.bat
> .\vcpkg integrate install
```

#### 步骤二：采用vcpkg安装eigen3、suitesparse、fmt依赖（耗时很长）

```
> .\vcpkg install eigen3:x64-windows
> .\vcpkg install suitesparse:x64-windows
> .\vcpkg install fmt:x64-windows
```

#### 步骤三：编译本仓库

```
> git clone https://github.com/zhanggefan/trajectory_regularizer.git
> cd motreg
```

这一步耗时很长，因为sophus依赖库代码有300MB大小

```
> git submodule update --init --recursive
```

如果希望编译python接口以便使用`examples/demo.py`，则首先应当保证pybind11正确安装：

```
> pip install pybind11
```

然后开始CMake编译：

```
> mkdir build
> cd build
> cmake -DPython3_EXECUTABLE={你的python路径}/python.exe -DCMAKE_TOOLCHAIN_FILE={你的vcpkg仓库路径}/scripts/buildsystems/vcpkg.cmake -S .. 
> cmake --build . --config Release
```

如果没有编译python接口的需要，则：

```
> mkdir build
> cd build
> cmake -DCMAKE_TOOLCHAIN_FILE={你的vcpkg仓库路径}/scripts/buildsystems/vcpkg.cmake -S .. 
> cmake --build . --config Release
```

编译得到的dll会落在```./motreg/bins/Release```文件夹下面

### Ubuntu下的编译过程

#### 步骤一：安装eigen3、suitesparse、fmt依赖

```
> sudo apt install libeigen3-dev
> sudo apt install libsuitesparse-dev
> sudo apt install libfmt-dev
```

#### 步骤二：编译本仓库

```
> git clone https://github.com/zhanggefan/trajectory_regularizer.git
> cd motreg
```

这一步耗时很长，因为sophus依赖库代码有300MB大小

```
> git submodule update --init --recursive
```

如果希望编译python接口以便使用`examples/demo.py`，则首先应当保证pybind11正确安装：

```
> pip install pybind11
```

然后开始CMake编译：

```
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE={你的python路径}/python -S .. 
> cmake --build .
```

如果没有编译python接口的需要，则：

```
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release -S .. 
> cmake --build .
```

编译得到的so会落在```./motreg/libs```文件夹下面

## Demo

注：Demo需要编译python接口。编译方法见上一章节   
安装所需要的python包

```
> pip install open3d
> pip install matplotlib
> pip install scipy
```

在本仓库目录下，配置`PYTHONPATH`环境变量并运行Demo

### Windows下运行Demo

```
> $env:PYTHONPATH="."
> python examples/demo.py
```

或运行依赖最小化的C++ API接口的Demo

```
> $env:PYTHONPATH="."
> python examples/demo_capi.py
```

### Ubuntu下运行Demo

```
> PYTHONPATH=. python examples/demo.py
```

或运行依赖最小化的C++ API接口的Demo

```
> PYTHONPATH=. python examples/demo_capi.py
```
