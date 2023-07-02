/*
 * Licensed to SpinorML under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * SpinorML licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::ffi::{c_void, CString};
use std::mem::zeroed;

use crate::cuda::*;
use crate::nvrtc::*;

pub type CudaDevice = CUdevice;
pub type CudaContext = CUcontext;
pub type CudaModule = CUmodule;
pub type CudaFunction = CUfunction;
pub type CudaStream = CUstream;
pub type CudaMemory = CUdeviceptr;

pub struct Driver;

impl Driver {
    pub fn init(device_number: u32) -> Result<(), &'static str> {
        let cu_result = unsafe { cuInit(device_number) };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuInit");
        }

        Ok(())
    }

    pub fn get_device(device_number: u32) -> Result<CudaDevice, &'static str> {
        let mut device = unsafe { zeroed::<CUdevice>() };
        let cu_result = unsafe { cuDeviceGet(&mut device as *mut CUdevice, device_number as i32) };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuDeviceGet");
        }

        Ok(device)
    }

    pub fn create_context(device: CudaDevice) -> Result<CudaContext, &'static str> {
        let mut context = unsafe { zeroed::<CUcontext>() };

        let cu_result = unsafe { cuCtxCreate_v2(&mut context as *mut CUcontext, 0, device) };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuCtxCreate_v2");
        }

        Ok(context)
    }

    pub fn load_module(ptx: CudaPtx) -> Result<CudaModule, &'static str> {
        let mut module = unsafe { zeroed::<CUmodule>() };

        let cu_result = unsafe {
            cuModuleLoadDataEx(
                &mut module as *mut CUmodule,
                ptx.as_ptr() as *const c_void,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            println!("Error: {}", cu_result);
            return Err("Failed: cuModuleLoadDataEx");
        }

        Ok(module)
    }

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    ///
    /// # Safety
    ///
    /// .
    pub unsafe fn get_function(
        module: CudaModule,
        name: &str,
    ) -> Result<CudaFunction, &'static str> {
        let mut kernel = zeroed::<CUfunction>();
        let name_str = CString::new(name).unwrap();

        let cu_result =
            cuModuleGetFunction(&mut kernel as *mut CUfunction, module, name_str.as_ptr());

        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuModuleGetFunction");
        }

        Ok(kernel)
    }

    pub fn create_stream() -> Result<CudaStream, &'static str> {
        let mut stream = unsafe { zeroed::<CUstream>() };

        let cu_result = unsafe { cuStreamCreate(&mut stream as *mut CUstream, 0) };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuStreamCreate");
        }

        Ok(stream)
    }

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    ///
    /// # Safety
    ///
    /// .
    pub unsafe fn launch_kernel(
        kernel: CudaFunction,
        num_blocks: (u32, u32, u32),
        num_threads: (u32, u32, u32),
        shared_memory_bytes: u32,
        stream: CudaStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> Result<(), &'static str> {
        let cu_result = cuLaunchKernel(
            kernel,
            num_blocks.0,
            num_blocks.1,
            num_blocks.2,
            num_threads.0,
            num_threads.1,
            num_threads.2,
            shared_memory_bytes,
            stream,
            kernel_params,
            extra,
        );
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuLaunchKernel");
        }

        Ok(())
    }

    pub fn allocate_memory(size: usize) -> Result<CudaMemory, &'static str> {
        let mut device_ptr = unsafe { zeroed::<CUdeviceptr>() };

        let cu_result = unsafe { cuMemAlloc_v2(&mut device_ptr as *mut CUdeviceptr, size) };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuMemAlloc_v2");
        }

        Ok(device_ptr)
    }

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    ///
    /// # Safety
    ///
    /// .
    pub unsafe fn copy_to_device(
        device_memory: CudaMemory,
        host_memory: *const c_void,
        size: usize,
    ) -> Result<(), &'static str> {
        let cu_result = cuMemcpyHtoD_v2(device_memory, host_memory, size);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuMemcpyHtoD_v2");
        }

        Ok(())
    }

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    ///
    /// # Safety
    ///
    /// .
    pub unsafe fn copy_from_device(
        host_memory: *mut c_void,
        device_memory: CudaMemory,
        size: usize,
    ) -> Result<(), &'static str> {
        let cu_result = cuMemcpyDtoH_v2(host_memory, device_memory, size);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuMemcpyDtoH_v2");
        }

        Ok(())
    }

    pub fn synchronize_context() -> Result<(), &'static str> {
        let cu_result = unsafe { cuCtxSynchronize() };
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuCtxSynchronize");
        }

        Ok(())
    }
}
