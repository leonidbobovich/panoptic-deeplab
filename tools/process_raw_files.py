import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import onnxruntime
from onnx_to_pytorch_helper import get_input_and_output_sizes
from os import listdir
from os.path import isfile, join

def main():
    mypath = sys.argv[1] if len(sys.argv) == 2 and os.path.exists(sys.argv[1]) else sys.exit(0)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)

if __name__ == '__main__': # Execute when the module is not initialized from an import statement.
    main()
