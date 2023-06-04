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

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CString};
use std::mem::zeroed;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

type CudaProgram = nvrtcProgram;
type CudaPtx = Vec<c_char>;
type CudaModule = CUmodule;
type CudaFunction = CUfunction;
type CudaStream = CUstream;

pub unsafe fn compile_program(name: &str, source: &str) -> Result<CudaProgram, &'static str> {
    let mut program = zeroed::<nvrtcProgram>();

    let nvrtc_result = nvrtcCreateProgram(
        &mut program as *mut nvrtcProgram,
        CString::new(source).unwrap().as_c_str().as_ptr(),
        CString::new(name).unwrap().as_c_str().as_ptr(),
        0,
        std::ptr::null(),
        std::ptr::null(),
    );
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcCreateProgram");
    }

    let compile_opts = vec![];
    let nvrtc_result = nvrtcCompileProgram(
        program,
        0,
        compile_opts.as_ptr(),
    );
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        println!("Compile resuilt: {}", nvrtc_result);
        return Err("Failed: nvrtcCompileProgram");
    }

    return Ok(program);
}

pub unsafe fn destroy_program(program: &mut CudaProgram) -> Result<(), &'static str> {
    let nvrtc_result = nvrtcDestroyProgram(program);
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcDestroyProgram");
    }

    return Ok(());
}

pub unsafe fn get_ptx(program: CudaProgram) -> Result<CudaPtx, &'static str> {
    let mut ptx_size: usize = 0;
    let nvrtc_result = nvrtcGetPTXSize(program, &mut ptx_size as *mut usize);
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcGetPTXSize");
    }
    println!("PTX size: {}", ptx_size);

    let mut buffer: Vec<c_char> = vec![0; ptx_size];
    let nvrtc_result = nvrtcGetPTX(program, buffer.as_mut_ptr());
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcGetPTX");
    }

    // let nvrtc_result = nvrtcDestroyProgram((*program) as *mut nvrtcProgram);
    // if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
    //     return Err("Failed: nvrtcDestroyProgram");
    // }

    return Ok(buffer);
}

pub unsafe fn get_program_log(program: CudaProgram) -> Result<&'static str, &'static str> {
    let mut log_size: usize = 0;
    let nvrtc_result = nvrtcGetProgramLogSize(program, &mut log_size as *mut usize);
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcGetProgramLogSize");
    }

    let mut log: Vec<c_char> = vec![0; log_size];
    let nvrtc_result = nvrtcGetProgramLog(program, log.as_mut_ptr());
    if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
        return Err("Failed: nvrtcGetProgramLogSize");
    }

    println!("\n** Log: {:?}", log);
    return Ok("Unlnown");
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

#[cfg(test)]
mod tests {
    use std::ffi::{c_void, CStr};
    use std::mem::size_of;
    use std::ptr::null_mut;

    use super::*;

    /**
     * To see the debug output you must run - cargo test test_get_device_info -- --nocapture
     */
    #[test]
    fn test_get_device_info() {
        let mut device = CUdevice::default();
        let mut raw_name: [i8; 128] = [0; 128];

        unsafe {
            let cu_result = cuInit(0);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuDeviceGet(&mut device as *mut CUdevice, 0);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuDeviceGetName(&mut raw_name as *mut i8, 128, device);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let name: &[u8; 256] = std::mem::transmute(&raw_name);
            println!("Device name: {:?}", CStr::from_bytes_until_nul(name).unwrap());
        }
    }

    #[test]
    fn test_vector_add() {
        let mut device = CUdevice::default();
        let mut context = &mut CUctx_st { _unused: [] } as *mut CUctx_st;
        let vector_add_kernel = "
            extern \"C\" __global__ void vector_add(float *a, float *b, float *c) {
                size_t idx = threadIdx.x;
                c[idx] = a[idx] + b[idx];
            }
        ";

        unsafe {
            let cu_result = cuInit(0);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuDeviceGet(&mut device as *mut CUdevice, 0);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuCtxCreate_v2(&mut context as *mut CUcontext, 0, device);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let program = compile_program("vector_add_kernel", vector_add_kernel).unwrap();
            let ptx = get_ptx(program).unwrap();

            let buffer_size = size_of::<f32>() * 5;
            let hA = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let hB = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let hC = [2.0f32, 4.0, 6.0, 8.0, 10.0];

            let mut dA = zeroed::<CUdeviceptr>();
            let mut dB = zeroed::<CUdeviceptr>();
            let mut dC = zeroed::<CUdeviceptr>();

            let cu_result = cuMemAlloc_v2(&mut dA as *mut CUdeviceptr, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuMemAlloc_v2(&mut dB as *mut CUdeviceptr, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuMemAlloc_v2(&mut dC as *mut CUdeviceptr, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuMemcpyHtoD_v2(dA, hA.as_ptr() as *const c_void, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuMemcpyHtoD_v2(dB, hB.as_ptr() as *const c_void, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let module = load_module(ptx).unwrap();
            let kernel = get_function(module, "vector_add").unwrap();
            let stream = create_stream().unwrap();

            let numBlocks = (1, 1, 1);
            let numThreads = (5, 1, 1);
            let sharedMemBytes = 0;
            let args = [&mut dA, &mut dB, &mut dC].as_mut_ptr() as *mut *mut c_void;
            let extra = null_mut();

            let cu_result = cuLaunchKernel(
                kernel,
                numBlocks.0, numBlocks.1, numBlocks.2,
                numThreads.0, numThreads.1, numThreads.2,
                sharedMemBytes,
                stream,
                args,
                extra);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuCtxSynchronize();
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            let cu_result = cuMemcpyDtoH_v2(hC.as_ptr() as *mut c_void, dC, buffer_size);
            assert_eq!(cu_result, cudaError_enum_CUDA_SUCCESS);

            for i in 0..5 {
                assert_eq!(hC[i], hA[i] + hB[i]);
            }
        }
    }
}
