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
use std::os::macos::raw::stat;

use crate::{CudaContext, CudaDevice, CudaFunction, CudaMemory, CudaModule, CudaPtx, CudaStream};
use crate::cuda::*;

pub struct Driver;

impl Driver {
    pub unsafe fn init(device_number: u32) -> Result<(), &'static str> {
        let cu_result = cuInit(device_number);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuInit");
        }

        return Ok(());
    }

    pub unsafe fn get_device(device_number: u32) -> Result<CudaDevice, &'static str> {
        let mut device = zeroed::<CUdevice>();
        let cu_result = cuDeviceGet(&mut device as *mut CUdevice, device_number as i32);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuInit");
        }

        return Ok(device);
    }

    pub unsafe fn create_context(device: CudaDevice) -> Result<CudaContext, &'static str> {
        let mut context = zeroed::<CUcontext>();

        let cu_result = cuCtxCreate_v2(&mut context as *mut CUcontext, 0, device);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuInit");
        }

        return Ok(context);
    }

    pub unsafe fn load_module(ptx: CudaPtx) -> Result<CudaModule, &'static str> {
        let mut module = zeroed::<CUmodule>();

        let cu_result = cuModuleLoadDataEx(
            &mut module as *mut CUmodule,
            ptx.as_ptr() as *const c_void,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            println!("Error: {}", cu_result);
            return Err("Failed: cuModuleLoad");
        }

        return Ok(module);
    }

    pub unsafe fn get_function(module: CudaModule, name: &str) -> Result<CudaFunction, &'static str> {
        let mut kernel = zeroed::<CUfunction>();
        let name_str = CString::new(name).unwrap();

        let cu_result = cuModuleGetFunction(&mut kernel as *mut CUfunction, module, name_str.as_ptr());
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuModuleGetFunction");
        }

        return Ok(kernel);
    }

    pub unsafe fn create_stream() -> Result<CudaStream, &'static str> {
        let mut stream = zeroed::<CUstream>();

        let cu_result = cuStreamCreate(&mut stream as *mut CUstream, 0);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuModuleGetFunction");
        }

        return Ok(stream);
    }

    pub unsafe fn allocate_memory(size: usize) -> Result<CudaMemory, &'static str> {
        let mut device_ptr = zeroed::<CUdeviceptr>();

        let cu_result = cuMemAlloc_v2(&mut device_ptr as *mut CUdeviceptr, size);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuModuleGetFunction");
        }

        return Ok(device_ptr);
    }

    pub unsafe fn copy_to_device(deviceMemory: CudaMemory, hostMemory: *const c_void, size: usize) -> Result<(), &'static str> {
        let cu_result = cuMemcpyHtoD_v2(deviceMemory, hostMemory, buffer_size);
        if cu_result != cudaError_enum_CUDA_SUCCESS {
            return Err("Failed: cuModuleGetFunction");
        }

        return Ok(());
    }
}
