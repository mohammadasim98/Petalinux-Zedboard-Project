""" (c) Copyright 2019 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
"""

from ctypes import *
import cv2
import numpy as np
import n2cube, dputils
import os
import threading
import time
import sys

l = threading.Lock()

def RunDPU(kernel_0, kernel_2, img, count):
    """
    DPU run function
    kernel: dpu kernel
    img: image to be run
    count : test rounds count
    """
    ##################################
    # Task for DPU Kernel 0
    """Create DPU Tasks from DPU Kernel"""
    task = n2cube.dpuCreateTask(kernel_0, 0)
    dputils.dpuSetInputImage2(task, KERNEL_CONV_INPUT_0.encode('utf-8'), img)
    tensor = n2cube.dpuGetInputTensor(task, KERNEL_CONV_INPUT_0.encode('utf-8'))
	
    """Run DPU Task"""
    n2cube.dpuRunTask(task)
    n2cube.dpuEnableTaskProfile(task)
    tasktime_kernel_0 = n2cube.dpuGetTaskProfile(task)
    
    ##################################
    # Global Average Pooling CPU Kernel 1
    """Get the output tensor size, width, channel, and height from Task's output"""
    size = n2cube.dpuGetOutputTensorSize(task, KERNEL_CONV_OUTPUT_0.encode('utf-8'))
    height = n2cube.dpuGetOutputTensorHeight(task, KERNEL_CONV_OUTPUT_0.encode('utf-8'))
    width = n2cube.dpuGetOutputTensorWidth(task, KERNEL_CONV_OUTPUT_0.encode('utf-8')) 
    channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_CONV_OUTPUT_0.encode('utf-8'))
    tmp = [0 for i in range(size)]
    out = [0 for i in range(channel)]
    n2cube.dpuGetOutputTensorInHWCInt8(task, KERNEL_CONV_OUTPUT_0.encode('utf-8'), tmp, size)
    for i in range(channel):
        out[i] = int(sum(tmp[i*height*width:(i+1)*height*width])) # Only sum is enough
    """Destroy DPU Tasks & free resources"""
    n2cube.dpuDestroyTask(task)
    
    ##################################
    # Task for DPU Kernel 2  
    task = n2cube.dpuCreateTask(kernel_2, 0)
    n2cube.dpuSetInputTensorInHWCInt8(task, KERNEL_FC_INPUT_2.encode('utf-8'), out, channel)
    tensor = n2cube.dpuGetInputTensor(task, KERNEL_FC_INPUT_2.encode('utf-8'))

    """Run DPU Task"""
    n2cube.dpuRunTask(task)
    n2cube.dpuEnableTaskProfile(task)
    tasktime_kernel_2 = n2cube.dpuGetTaskProfile(task)

    ##################################
    # Softmax CPU Kernel 3
    size = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT_2.encode('utf-8'))
    softmax_FP32 = [0 for i in range(size)]
    softmax = [0 for i in range(size)]
    n2cube.dpuGetOutputTensorInHWCFP32(task, KERNEL_FC_OUTPUT_2.encode('utf-8'), softmax_FP32, size)

    """Print Results"""
    softmax = [np.exp(ele/(height*width)) for ele in softmax_FP32]
    softmax_sum = sum(softmax)
    softmax = [(ele/softmax_sum)  for ele in softmax]
    print("No DR: ", softmax[0])
    print("Mild Non-Proliferative DR: ", softmax[1])
    print("Moderate Non-Proliferative DR: ", softmax[2])
    print("Severe Non-Proliferative DR: ", softmax[3])
    print("Proliferative DR: ", softmax[4]) 
    print("\n Predicted Class: ", np.argmax(softmax))
    print("Total DPU Kernel execution time (us): ", tasktime_kernel_2 + tasktime_kernel_0 )
    print("\n\n")
    """Destroy DPU Task & free resources"""
    n2cube.dpuDestroyTask(task)
    
    l.acquire()
    count = count + threadnum
    l.release()

global threadnum
threadnum = 0
KERNEL_CONV_0 = "resnet50_0"
KERNEL_CONV_INPUT_0 = "resnet50_5_conv1_Conv2D"
KERNEL_CONV_OUTPUT_0 = "resnet50_5_res5c_branch2c_Conv2D"

KERNEL_FC_2 = "resnet50_2"
KERNEL_FC_INPUT_2 = "dense_4_MatMul"
KERNEL_FC_OUTPUT_2 = "dense_4_MatMul"

"""
brief Entry for runing Resnet50 neural network
"""
def main(argv):

    """Attach to DPU driver and prepare for runing"""
    n2cube.dpuOpen()

    image_path = "./../image_256_256/"
    
    listimage = os.listdir(image_path)
    for i in range(3):
        """Create DPU Kernels for GoogLeNet"""
        kernel_0 = n2cube.dpuLoadKernel(KERNEL_CONV_0.encode('utf-8'))
        kernel_2 = n2cube.dpuLoadKernel(KERNEL_FC_2.encode('utf-8'))
        path = os.path.join(image_path, listimage[i])
    
        print("Loading  %s" %listimage[i])
    
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        threadAll = []
        global threadnum
        threadnum = int(argv[1])
        print("Input thread number is: %d" %threadnum)
    
        time1 = time.time()
    
        for i in range(int(threadnum)):
            t1 = threading.Thread(target=RunDPU, args=(kernel_0, kernel_2, img, i))
            threadAll.append(t1)
        for x in threadAll:
            x.start()
        for x in threadAll:
            x.join()
    
        time2 = time.time()
    
        timetotal = time2 - time1
        fps = float(1000 / timetotal)
        print("%.2f FPS" %fps)
        print("Overall Execution Time", timetotal)
        """Destroy DPU Tasks & free resources"""
        rtn = n2cube.dpuDestroyKernel(kernel_0)
        rtn = n2cube.dpuDestroyKernel(kernel_2)

    """Dettach from DPU driver & release resources"""
    n2cube.dpuClose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please input thread number.")
    else :
        main(sys.argv)
