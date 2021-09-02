'''
-- (c) Copyright 2019 Xilinx, Inc. All rights reserved.
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
'''
from ctypes import *
import cv2
import numpy as np
import n2cube

try:
    pyc_libdputils = cdll.LoadLibrary("libdputils.so")
except Exception:
    print('Load libdputils.so failed\nPlease install DNNDK first!')


def dpuSetInputImageWithScale(task, nodeName, image, mean, scale, idx=0):
    """Set image into DPU Task's input Tensor with a specified scale parameter"""
    height = n2cube.dpuGetInputTensorHeight(task, nodeName, idx)
    width = n2cube.dpuGetInputTensorWidth(task, nodeName, idx)
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    (imageHeight, imageWidth, imageChannel) = image.shape
    inputMean = (c_float * channel)()
    for i in range(0, channel):
        inputMean[i] = mean[i]

    if height == imageHeight and width == imageWidth:
        newImage = image
    else:
        newImage = cv2.resize(image, (width, height), 0, 0, cv2.INTER_LINEAR)

    inputImage = np.asarray(newImage, dtype=np.byte)
    inputImage2 = inputImage.ctypes.data_as(c_char_p)
    return pyc_libdputils.pyc_dpuSetInputData(task,
                                              c_char_p(nodeName), inputImage2,
                                              c_int(height),
                                              c_int(width),
                                              c_int(imageChannel), inputMean,
                                              c_float(scale), c_int(idx))


def dpuSetInputImage(task, nodeName, image, mean, idx=0):
    """
    Set image into DPU Task's input Tensor
    task: DPU Task
    nodeName: The pointer to DPU Node name.
    image:    Input image in OpenCV Mat format. Single channel and 3-channel input image are supported.
    mean:     Mean value array which contains 1 member for single channel input image
              or 3 members for 3-channel input image
              Note: You can get the mean values from the input Caffe prototxt.
                    At present, the format of mean value file is not yet supported
    idx:      The index of a single input tensor for the Node, with default value as 0
    """
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    pyc_libdputils.paraCheck(task, c_char_p(nodeName), c_int(idx))
    inputMean = (c_float * channel)()
    for i in range(0, channel):
        inputMean[i] = mean[i]
    pyc_libdputils.inputMeanValueCheck(inputMean)
    return dpuSetInputImageWithScale(task, nodeName, image, mean, 1.0, idx)


def dpuSetInputImage2(task, nodeName, image, idx=0):
    """
    Set image into DPU Task's input Tensor (mean values automatically processed by N2Cube)
    nodeName: The pointer to DPU Node name.
    image:    Input image in OpenCV Mat format. Single channel and 3-channel input image are supported.
    idx:      The index of a single input tensor for the Node, with default value as 0
    """
    channel = n2cube.dpuGetInputTensorChannel(task, nodeName, idx)
    pyc_libdputils.paraCheck(task, c_char_p(nodeName), c_int(idx))
    output = create_string_buffer(sizeof(c_float) * (channel))
    outputMean = POINTER(c_float)(output)
    pyc_libdputils.paraCheck(task, c_char_p(nodeName), c_int(idx))
    pyc_libdputils.loadMean(task, outputMean, channel)
    for i in range(channel):
        outputMean[i] = float(outputMean[i])
    return dpuSetInputImageWithScale(task, nodeName, image, outputMean, 1.0,
                                     idx)
