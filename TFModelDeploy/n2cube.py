"""
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
"""
from ctypes import *
import numpy as np

try:
    pyc_libn2cube = cdll.LoadLibrary("libn2cube.so")
except Exception:
    print('Load libn2cube.so failed\nPlease install DNNDK first!')


def dpuOpen():
    """
    Open & initialize the usage of DPU device
    Returns: 0 on success, or negative value in case of failure.
             Error message (Fail to open DPU device) is reported if any error takes place
    """
    return pyc_libn2cube.pyc_dpuOpen()


def dpuClose():
    """
    Close & finalize the usage of DPU devicei
    Returns: 0 on success, or negative error ID in case of failure.
             Error message (Fail to close DPU device) is reported if any error takes place
    """
    return pyc_libn2cube.pyc_dpuClose()


def dpuLoadKernel(kernelName):
    """
    Load a DPU Kernel and allocate DPU memory space for
    its Code/Weight/Bias segments
    kernelName: The pointer to neural network name.
                Use the names produced by Deep Neural Network Compiler (DNNC) after
                the compilation of neural network.
                For each DL application, perhaps there are many DPU Kernels existing
                in its hybrid CPU+DPU binary executable. For each DPU Kernel, it has
                one unique name for differentiation purpose
    Returns: The loaded DPU Kernel on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuLoadKernel.restype = POINTER(c_void_p)
    return pyc_libn2cube.pyc_dpuLoadKernel(c_char_p(kernelName))


def dpuDestroyKernel(kernel):
    """
    Destroy a DPU Kernel and release its associated resources
    kernel:  The DPU Kernel to be destroyed. This parameter should be gotten from the result of dpuLoadKernel()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuDestroyKernel(kernel)


def dpuCreateTask(kernel, mode):
    """
    Instantiate a DPU Task from one DPU Kernel, allocate its private
    working memory buffer and prepare for its execution context
    kernel:  The DPU Kernel. This parameter should be gotten from the result of dpuLoadKernel()
    mode:    The running mode of DPU Task. There are 3 available modes:
               MODE_NORMAL: default mode identical to the mode value 0.
               MODE_PROF: output profiling information layer by layer while running of DPU Task,
                     which is useful for performance analysis.
               MODE_DUMP: dump the raw data for DPU Task's CODE/BIAS/WEIGHT/INPUT/OUTPUT layer by layer
    Returns: 0 on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuCreateTask.restype = POINTER(c_void_p)
    return pyc_libn2cube.pyc_dpuCreateTask(kernel, c_int(mode))


def dpuDestroyTask(task):
    """
    Remove a DPU Task, release its working memory buffer and destroy
    associated execution context
    task:    DPU Task. This parameter should be gotten from the result of dpuCreatTask()
    Returns: 0 on success, or negative value in case of any failure
    """
    return pyc_libn2cube.pyc_dpuDestroyTask(task)


def dpuRunTask(task):
    """
    Launch the running of DPU Task
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or negative value in case of any failure
    """
    return pyc_libn2cube.pyc_dpuRunTask(task)


def dpuEnableTaskDebug(task):
    """
    Enable dump facility of DPU Task while running for debugging purpose
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuEnableTaskDebug(task)


def dpuEnableTaskProfile(task):
    """
    Enable profiling facility of DPU Task while running to get its performance metrics
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: 0 on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuEnableTaskProfile(task)


def dpuGetTaskProfile(task):
    """
    Get the execution time of DPU Task
    task:    DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    Returns: The DPU Task's execution time (us) after its running
    """
    pyc_libn2cube.pyc_dpuGetTaskProfile.restype = c_longlong
    return pyc_libn2cube.pyc_dpuGetTaskProfile(task)


def dpuGetNodeProfile(task, nodeName):
    """
    Get the execution time of DPU Node
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    Returns:  The DPU Node's execution time (us) after its running
    """
    pyc_libn2cube.pyc_dpuGetNodeProfile.restype = c_longlong
    return pyc_libn2cube.pyc_dpuGetNodeProfile(task, c_char_p(nodeName))


"""
API for both single IO and multiple IO.
For multiply IO, should specify the input/output tensor idx.
"""


def dpuGetInputTensorCnt(task, nodeName):
    """
    Get total number of input Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The total number of input tensor for specified Node.
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorCnt(task, c_char_p(nodeName))


def dpuGetInputTensor(task, nodeName, idx=0):
    """
    Get input Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The pointer to Task input Tensor on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuGetInputTensor.restype = POINTER(c_void_p)
    return pyc_libn2cube.pyc_dpuGetInputTensor(task,
                                               c_char_p(nodeName), c_int(idx))


def dpuGetTensorData(tensorAddress, data, tensorSize):
    """
    Get the tensor data from the address that returnd by dpuGetOutputTensorAddress
    tensorAddress: Result from dpuGetOutputTensorAddress()
    data:          Output data
    tensorSize:    Size of the output data
    Returns:       Data output.
    """
    for i in range(tensorSize):
        data[i] = int(tensorAddress[i])
    return


def dpuGetInputTensorSize(task, nodeName, idx=0):
    """
    Get the size (in byte) of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The size of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorSize(
        task, c_char_p(nodeName), c_int(idx))


def dpuGetInputTensorScale(task, nodeName, idx=0):
    """
    Get the scale value (DPU INT8 quantization) of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Retruns:  The scale value of Task's input Tensor on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuGetInputTensorScale.restype = c_float
    return pyc_libn2cube.pyc_dpuGetInputTensorScale(task,
                                                    c_char_p(nodeName),
                                                    c_int(idx))


def dpuGetInputTensorHeight(task, nodeName, idx=0):
    """
    Get the height dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The height dimension of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorHeight(task,
                                                     c_char_p(nodeName),
                                                     c_int(idx))


def dpuGetInputTensorWidth(task, nodeName, idx=0):
    """
    Get the width dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The width dimension of Task's input Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorWidth(task,
                                                    c_char_p(nodeName),
                                                    c_int(idx))


def dpuGetInputTensorChannel(task, nodeName, idx=0):
    """
    Get the channel dimension of one DPU Task input Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The channel dimension of Task's input Tensor on success, or report error in case of any failure.
    """
    return pyc_libn2cube.pyc_dpuGetInputTensorChannel(task,
                                                      c_char_p(nodeName),
                                                      c_int(idx))


def dpuGetOutputTensorCnt(task, nodeName):
    """
    Get total number of output Tensor of DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    Returns:  The total number of output tensor for the DPU Task
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorCnt(task, c_char_p(nodeName))


def dpuGetOutputTensor(task, nodeName, idx=0):
    """
    Get output Tensor of one DPU Task
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The pointer to Task's output Tensor on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuGetOutputTensor.restype = POINTER(c_void_p)
    return pyc_libn2cube.pyc_dpuGetOutputTensor(task,
                                                c_char_p(nodeName), c_int(idx))


def dpuGetOutputTensorSize(task, nodeName, idx=0):
    """
    Get the size (in byte) of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The size of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorSize(task,
                                                    c_char_p(nodeName),
                                                    c_int(idx))


def dpuGetOutputTensorAddress(task, nodeName, idx=0):
    """
    Get the start address of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The start addresses to Task's output Tensor on success, or report error in case of any failure 
    """
    size = dpuGetOutputTensorSize(task, nodeName, idx)
    output = create_string_buffer(sizeof(c_byte) * size)
    outputPP = POINTER(c_byte)(output)
    pyc_libn2cube.pyc_dpuGetOutputTensorAddress.restype = POINTER(c_byte)
    outputPP = pyc_libn2cube.pyc_dpuGetOutputTensorAddress(task,
                                                           c_char_p(nodeName),
                                                           c_int(idx))
    return outputPP


def dpuGetOutputTensorScale(task, nodeName, idx=0):
    """
    Get the scale value (DPU INT8 quantization) of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The scale value of Task's output Tensor on success, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuGetOutputTensorScale.restype = c_float
    return pyc_libn2cube.pyc_dpuGetOutputTensorScale(task,
                                                     c_char_p(nodeName),
                                                     c_int(idx))


def dpuGetOutputTensorHeight(task, nodeName, idx=0):
    """
    Get the height dimension of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The height dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorHeight(task,
                                                      c_char_p(nodeName),
                                                      c_int(idx))


def dpuGetOutputTensorWidth(task, nodeName, idx=0):
    """
    Get the channel dimension of one DPU Task output Tensor
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The width dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorWidth(task,
                                                     c_char_p(nodeName),
                                                     c_int(idx))


def dpuGetOutputTensorChannel(task, nodeName, idx=0):
    """
    Get DPU Node's output tensor's channel
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  The channel dimension of Task's output Tensor on success, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetOutputTensorChannel(task,
                                                       c_char_p(nodeName),
                                                       c_int(idx))


def dpuGetTensorSize(tensor):
    """
    Get the size of one DPU Tensor
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The size of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorSize(tensor)


def dpuGetTensorAddress(tensor):
    """
    Get the address of one DPU tensor
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The pointer of Tensor list, or report error in case of any failure
    """#TODO
    size = dpuGetTensorSize(tensor)
    output = create_string_buffer(sizeof(c_byte) * size)
    outputPP = POINTER(c_byte)(output)
    pyc_libn2cube.pyc_dpuGetTensorAddress.restype = POINTER(c_byte)
    outputPP = pyc_libn2cube.pyc_dpuGetTensorAddress(tensor)
    return outputPP


def dpuGetTensorScale(tensor):
    """
    Get the scale value of one DPU Tensor
    Returns: The scale value of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The scale value of Tensor, or report error in case of any failure
    """
    pyc_libn2cube.pyc_dpuGetTensorScale.restype = c_float
    return pyc_libn2cube.pyc_dpuGetTensorScale(tensor)


def dpuGetTensorHeight(tensor):
    """
    Get the height dimension of one DPU Tensor
    Returns: The height dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The height dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorHeight(tensor)


def dpuGetTensorWidth(tensor):
    """
    Get the width dimension of one DPU Tensor
    Returns: The width dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The width dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorWidth(tensor)


def dpuGetTensorChannel(tensor):
    """
    Get the channel dimension of one DPU Tensor
    Returns: The channel dimension of Tensor, or report error in case of any failure
    tensor:  DPU tensor. This parameter should be gotten from the reslut of dpuGetOutputTensor()
    Returns: The channel dimension of Tensor, or report error in case of any failure
    """
    return pyc_libn2cube.pyc_dpuGetTensorChannel(tensor)


def dpuSetInputTensorInCHWInt8(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under Caffe
    Blob's order (channel/height/width) in INT8 format 
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    inputData = (c_byte * size)()
    for i in range(0, size):
        inputData[i] = data[i]
    return pyc_libn2cube.pyc_dpuSetInputTensorInCHWInt8(task,
                                                        c_char_p(nodeName),
                                                        inputData,
                                                        c_int(size), c_int(idx))


def dpuSetInputTensorInCHWFP32(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under Caffe
    Blob's order (channel/height/width) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    inputData = (c_float * size)()
    for i in range(0, size):
        inputData[i] = data[i]
    return pyc_libn2cube.pyc_dpuSetInputTensorInCHWFP32(task,
                                                        c_char_p(nodeName),
                                                        inputData,
                                                        c_int(size), c_int(idx))


def dpuSetInputTensorInHWCInt8(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under DPU
    Tensor's order (height/width/channel) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    inputData = (c_byte * size)()
    for i in range(0, size):
        inputData[i] = data[i]
    return pyc_libn2cube.pyc_dpuSetInputTensorInHWCInt8(task,
                                                        c_char_p(nodeName),
                                                        inputData,
                                                        c_int(size), c_int(idx))


def dpuSetInputTensorInHWCFP32(task, nodeName, data, size, idx=0):
    """
    Set DPU Task's input Tensor with data stored under DPU
    Tensor's order (height/width/channel) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The input data
    size:     The size (in Bytes) of input data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    inputData = (c_float * size)()
    for i in range(0, size):
        inputData[i] = data[i]
    pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32.argtypes = (
        POINTER(c_void_p), c_char_p, POINTER(c_float), c_int, c_int)
    return pyc_libn2cube.pyc_dpuSetInputTensorInHWCFP32(task,
                                                        c_char_p(nodeName),
                                                        inputData,
                                                        c_int(size), c_int(idx))


def dpuGetOutputTensorInCHWInt8(task, nodeName, data, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under Caffe
    Blob's order (channel/height/width) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The output Tensor data
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    output = create_string_buffer(sizeof(c_byte) * size)
    outputP = POINTER(c_byte)(output)
    pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8.argtypes = (
        POINTER(c_void_p), c_char_p, POINTER(c_byte), c_int, c_int)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWInt8(task,
                                                        c_char_p(nodeName),
                                                        outputP,
                                                        c_int(size), c_int(idx))
    for i in range(size):
        data[i] = int(outputP[i])
    return rtn


def dpuGetOutputTensorInCHWFP32(task, nodeName, data, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under Caffe
    Blob's order (channel/height/width) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The output Tensor's data
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    output = create_string_buffer(sizeof(c_float) * size)
    outputP = POINTER(c_float)(output)
    pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32.argtypes = (
        POINTER(c_void_p), c_char_p, POINTER(c_float), c_int, c_int)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInCHWFP32(task,
                                                        c_char_p(nodeName),
                                                        outputP,
                                                        c_int(size), c_int(idx))
    for i in range(size):
        data[i] = float(outputP[i])
    return rtn


def dpuGetOutputTensorInHWCInt8(task, nodeName, data, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under DPU
    Tensor's order (height/width/channel) in INT8 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The output Tensor's data
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    output = create_string_buffer(sizeof(c_byte) * size)
    outputP = POINTER(c_byte)(output)
    pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8.argtypes = (
        POINTER(c_void_p), c_char_p, POINTER(c_byte), c_int, c_int)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCInt8(task,
                                                        c_char_p(nodeName),
                                                        outputP,
                                                        c_int(size), c_int(idx))
    for i in range(size):
        data[i] = int(outputP[i])
    return rtn


def dpuGetOutputTensorInHWCFP32(task, nodeName, data, size, idx=0):
    """
    Get DPU Task's output Tensor and store them under DPU
    Tensor's order (height/width/channel) in FP32 format
    task:     DPU Task. This parameter should be gotten from the result of  dpuCreatTask()
    nodeName: The pointer to DPU Node's name
    data:     The output Tensor's data
    size:     The size (in Bytes) of output data to be stored
    idx:      The index of a single output tensor for the Node, with default value of 0
    Returns:  0 on success, or report error in case of failure
    """
    output = create_string_buffer(sizeof(c_float) * size)
    outputP = POINTER(c_float)(output)
    pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32.argtypes = (
        POINTER(c_void_p), c_char_p, POINTER(c_float), c_int, c_int)
    rtn = pyc_libn2cube.pyc_dpuGetOutputTensorInHWCFP32(task,
                                                        c_char_p(nodeName),
                                                        outputP,
                                                        c_int(size), c_int(idx))
    for i in range(size):
        data[i] = float(outputP[i])
    return rtn


def dpuRunSoftmax(inputData, outputData, numClasses, batchSize, scale):
    """
    Compute softmax
    inputData:  Softmax input.
                This parameter should be gotten from the result of  dpuGetOuputTensorAddress()
    outputData: Result of softmax
    numClasses: The number of classes that softmax calculation operates on
    batchSize:  Batch size for the softmax calculation.
                This parameter should be specified with the division of the element number by inputs by numClasses
    scale:      The scale value applied to the input elements before softmax calculation
                This parameter typically can be obtained by using DNNDK API dpuGetRensorScale()
    Returns:    0 on success, or report error in case of failure
    """
    output = create_string_buffer(sizeof(c_float) * (numClasses * batchSize))
    outputP = POINTER(c_float)(output)
    pyc_libn2cube.pyc_dpuRunSoftmax.argtypes = (
        POINTER(c_byte), POINTER(c_float), c_int, c_int, c_float)
    rtn = pyc_libn2cube.pyc_dpuRunSoftmax(inputData, outputP,
                                          c_int(numClasses),
                                          c_int(batchSize), c_float(scale))
    for i in range(numClasses * batchSize):
        outputData[i] = float(outputP[i])
    return rtn


def dpuSetExceptionMode(mode):
    """
    Set the exception handling mode for DNNDK runtime N2Cube.
    It will affect all the APIs included in the libn2cube library
    mode:    The exception handling mode for runtime N2Cube to be specified.
             Available values include:
              -   N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT
              -   N2CUBE_EXCEPTION_MODE_RET_ERR_CODE
    Returns: 0 on success, or negative value in case of failure
    """
    return pyc_libn2cube.pyc_dpuSetExceptionMode(c_int(mode))


def dpuGetExceptionMode():
    """
    Get the exception handling mode for runtime N2Cube
    Returns: Current exception handing mode for N2Cube APIs.
             Available values include:
             -   N2CUBE_EXCEPTION_MODE_PRINT_AND_EXIT
             -   N2CUBE_EXCEPTION_MODE_RET_ERR_CODE
    """
    return pyc_libn2cube.pyc_dpuGetExceptionMode()


def dpuGetExceptionMessage(error_code):
    """
    Get the error message from error code (always negative value) returned by N2Cube APIs
    Returns: A pointer to a const string, indicating the error message for error_code
    """
    pyc_libn2cube.pyc_dpuGetExceptionMessage.restype = POINTER(c_char)
    return pyc_libn2cube.pyc_dpuGetExceptionMessage(c_int(error_code))
