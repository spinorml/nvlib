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

use std::ffi::{c_void};
use std::mem::size_of;

use nvrust::nvrtc::Nvrtc;
use nvrust::driver::Driver;

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
        let _context = Driver::create_context(device).unwrap();

        let program = Nvrtc::compile_program("vector_add_kernel", vector_add_kernel).unwrap();
        let ptx = Nvrtc::get_ptx(program).unwrap();

        let buffer_size = size_of::<f32>() * 5;
        let h_a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let h_b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let h_c = [0.0f32, 0.0, 0.0, 0.0, 0.0];

        let mut d_a = Driver::allocate_memory(buffer_size).unwrap();
        let mut d_b = Driver::allocate_memory(buffer_size).unwrap();
        let mut d_c = Driver::allocate_memory(buffer_size).unwrap();

        Driver::copy_to_device(d_a, h_a.as_ptr() as *const c_void, buffer_size).unwrap();
        Driver::copy_to_device(d_b, h_b.as_ptr() as *const c_void, buffer_size).unwrap();

        let module = Driver::load_module(ptx).unwrap();
        let kernel = Driver::get_function(module, "vector_add").unwrap();
        let stream = Driver::create_stream().unwrap();

        let num_blocks = (1, 1, 1);
        let num_threads = (5, 1, 1);
        let shared_mem_bytes = 0;
        let kernel_params = [&mut d_a, &mut d_b, &mut d_c].as_ptr() as *mut *mut c_void;
        let extra = std::ptr::null_mut();

        Driver::launch_kernel(
            kernel,
            num_blocks,
            num_threads,
            shared_mem_bytes,
            stream,
            kernel_params,
            extra).unwrap();

        Driver::synchronize_context().unwrap();

        Driver::copy_from_device(h_c.as_ptr() as *mut c_void, d_c, buffer_size).unwrap();

        assert_eq!(h_c, [2.0f32, 4.0, 6.0, 8.0, 10.0]);
    }
}