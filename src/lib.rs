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

pub mod cuda {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod nvrtc;
pub mod driver;

use std::ffi::{c_char, c_void, CString};
use std::mem::zeroed;

use crate::driver::*;
use crate::nvrtc::*;
use crate::cuda::*;

type CudaDevice = CUdevice;
type CudaContext = CUcontext;
type CudaProgram = nvrtcProgram;
type CudaPtx = Vec<c_char>;
type CudaModule = CUmodule;
type CudaFunction = CUfunction;
type CudaStream = CUstream;
type CudaMemory = CUdeviceptr;

#[cfg(test)]
mod tests {
    use std::ffi::{c_void, CStr};
    use std::mem::size_of;
    use std::ptr::null_mut;

    use super::*;

    /**
     * To see the debug output you must run - cargo test test_get_device_info -- --nocapture
     */
    // #[test]
    // fn test_get_device_info() {
    //     let mut device = CUdevice::default();
    //     let mut raw_name: [i8; 128] = [0; 128];
    //
    //     unsafe {
    //         let cu_result = cuInit(0);
    //         assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
    //
    //         let cu_result = cuDeviceGet(&mut device as *mut CUdevice, 0);
    //         assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
    //
    //         let cu_result = cuDeviceGetName(&mut raw_name as *mut i8, 128, device);
    //         assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
    //
    //         let name: &[u8; 256] = std::mem::transmute(&raw_name);
    //         println!("Device name: {:?}", CStr::from_bytes_until_nul(name).unwrap());
    //     }
    // }

    #[test]
    fn test_vector_add() {
        // let mut device = CUdevice::default();
        // let mut context = &mut CUctx_st { _unused: [] } as *mut CUctx_st;
        let vector_add_kernel = "
            extern \"C\" __global__ void vector_add(float *a, float *b, float *c) {
                size_t idx = threadIdx.x;
                c[idx] = a[idx] + b[idx];
            }
        ";

        unsafe {
            Driver::init(0).unwrap();
            let device = Driver::get_device(0).unwrap();
            let context = Driver::create_context(device).unwrap();

            let program = Nvrtc::compile_program("vector_add_kernel", vector_add_kernel).unwrap();

            let buffer_size = size_of::<f32>() * 5;
            let hA = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let hB = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let hC = [2.0f32, 4.0, 6.0, 8.0, 10.0];

            let mut dA = Driver::allocate_memory(buffer_size).unwrap();
            let mut dB = Driver::allocate_memory(buffer_size).unwrap();
            let mut dC = Driver::allocate_memory(buffer_size).unwrap();

            Driver::copy_to_device(dA, hA.as_ptr() as *const c_void, buffer_size).unwrap();
            Driver::copy_to_device(dB, hB.as_ptr() as *const c_void, buffer_size).unwrap();

            //
            // let cu_result = cuMemcpyHtoD_v2(dA, hA.as_ptr() as *const c_void, buffer_size);
            // assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
            //
            // let cu_result = cuMemcpyHtoD_v2(dB, hB.as_ptr() as *const c_void, buffer_size);
            // assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
            //
            // let module = load_module(ptx).unwrap();
            // let kernel = get_function(module, "vector_add").unwrap();
            // let stream = create_stream().unwrap();
            //
            // let numBlocks = (1, 1, 1);
            // let numThreads = (5, 1, 1);
            // let sharedMemBytes = 0;
            // let args = [&mut dA, &mut dB, &mut dC].as_mut_ptr() as *mut *mut c_void;
            // let extra = null_mut();
            //
            // let cu_result = cuLaunchKernel(
            //     kernel,
            //     numBlocks.0, numBlocks.1, numBlocks.2,
            //     numThreads.0, numThreads.1, numThreads.2,
            //     sharedMemBytes,
            //     stream,
            //     args,
            //     extra);
            // assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
            //
            // let cu_result = cuCtxSynchronize();
            // assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
            //
            // let cu_result = cuMemcpyDtoH_v2(hC.as_ptr() as *mut c_void, dC, buffer_size);
            // assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);
            //
            // for i in 0..5 {
            //     assert_eq!(hC[i], hA[i] + hB[i]);
            // }
        }
    }
}
