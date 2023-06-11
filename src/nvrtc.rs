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

use std::ffi::{c_char, CString};
use std::mem::zeroed;

use crate::cuda::*;

pub type CudaProgram = nvrtcProgram;
pub type CudaPtx = Vec<c_char>;

pub struct Nvrtc;

impl Nvrtc {
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

        return Ok(buffer);
    }

    pub unsafe fn destroy_program(mut program: CudaProgram) -> Result<(), &'static str> {
        let nvrtc_result = nvrtcDestroyProgram(&mut program as *mut nvrtcProgram);
        if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
            return Err("Failed: nvrtcDestroyProgram");
        }

        return Ok(());
    }

    pub unsafe fn get_program_log(program: CudaProgram) -> Result<String, &'static str> {
        let mut log_size: usize = 0;
        let nvrtc_result = nvrtcGetProgramLogSize(program, &mut log_size as *mut usize);
        if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
            return Err("Failed: nvrtcGetProgramLogSize");
        }

        let mut raw_log: Vec<u8> = vec![0; log_size];
        let nvrtc_result = nvrtcGetProgramLog(program, raw_log.as_mut_ptr() as *mut c_char);
        if nvrtc_result != nvrtcResult_NVRTC_SUCCESS {
            return Err("Failed: nvrtcGetProgramLog");
        }

        return Ok(String::from_utf8(raw_log).unwrap());
    }
}
