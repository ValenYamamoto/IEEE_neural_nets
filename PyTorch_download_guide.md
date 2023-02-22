# PyTorch Download Guide

## (Prerequiste) Download Python
You can download the latest version of Python [here](https://python.org/downloads). Simply download the installer and run the executable file.

PyTorch only supports Python 3.x, though Python 3.5 or greater is recomended. The latest stable version of Python at the time of writing this is 3.8.5, which is fine for PyTorch.

With Python, you should have also downloaded pip. Check in your Python/Scripts directory for the pip.exe (or pip3.exe) file. 
If pip is not there (unlikely), you will have to download pip yourself by going [here](https://pypi.org/project/pip/), which will give you a python file that will give you pip. Run the file in your Python/Scripts directory.

## Installing PyTorch
In this tutorial, we are not using GPUs. I used this pip command to download PyTorch, which should work for both **Windows** and **Linux**.
```bash
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
For Macs, this simpler command will suffice:
```bash
pip install torch torchvision
```
For more information on installation and for different installation configurations (for example, with CUDA support), see PyTorch's download page [here](https://pytorch.org/get-started/locally/). This page also has sample code to run to test whether all packages have been installed correctly.

## (Optional) Install NumPy
Many of the early examples are done in NumPy. In order to experiment with the examples, the NumPy library needs to be installed. This can be easily done with pip:
```bash
pip install numpy
```
