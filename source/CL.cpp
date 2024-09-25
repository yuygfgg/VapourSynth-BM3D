#include<OpenCL/opencl.h>
#include <iostream>
#include <vector>
#include "include/Block.h"

// OpenCL 相关的全局变量
cl_platform_id g_platform;
cl_device_id g_device;
cl_context g_context;
cl_command_queue g_queue;
cl_program g_program;
cl_kernel g_kernel;
bool g_opencl_initialized = false;

// 初始化 OpenCL 资源
void initOpenCL() {
    if (g_opencl_initialized) return;  // 如果已经初始化，则不再重复初始化

    cl_int err;

    // 获取平台 ID
    err = clGetPlatformIDs(1, &g_platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting platform ID: " << err << std::endl;
        exit(1);
    }

    // 获取设备 ID
    err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting device ID: " << err << std::endl;
        exit(1);
    }

    // 创建上下文
    g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating context: " << err << std::endl;
        exit(1);
    }

    // 创建命令队列
    g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating command queue: " << err << std::endl;
        exit(1);
    }

    // 创建 OpenCL 程序
    const char* kernel_source = R"CLC(
        #define MAX_BLOCK_SIZE 256  // Set this to the maximum expected block size

        __kernel void block_matching_multi(
            __global const float* ref_block,
            __global const float* src_image,
            __global const int2* search_pos,
            const int block_width,
            const int block_height,
            const int src_stride,
            const int src_range,
            const float thSSE,
            const float distMul,
            __global float* output,
            __global int2* output_pos
        ) {
            int idx = get_global_id(0);
            int2 pos = search_pos[idx];

            __local float local_ref_block[MAX_BLOCK_SIZE];

            // Load reference block into local memory
            int block_size = block_width * block_height;
            for (int i = get_local_id(0); i < block_size; i += get_local_size(0)) {
                local_ref_block[i] = ref_block[i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const __global float* srcp0 = src_image + pos.y * src_stride + pos.x;
            float dist = 0.0f;

            // Vectorized loop
            for (int y = 0; y < block_height; ++y) {
                for (int x = 0; x < block_width; x += 4) {
                    float4 ref_val = vload4(x, &local_ref_block[y * block_width]);
                    float4 src_val = vload4(x, srcp0 + y * src_stride);
                    float4 diff = ref_val - src_val;
                    float4 diff_sq = diff * diff;
                    dist += diff_sq.s0 + diff_sq.s1 + diff_sq.s2 + diff_sq.s3;
                }
            }

            if (dist <= thSSE && dist != 0) {
                output[idx] = dist * distMul;
                output_pos[idx] = pos;
            } else {
                output[idx] = -1.0f;
            }
        })CLC";

    g_program = clCreateProgramWithSource(g_context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating program: " << err << std::endl;
        exit(1);
    }

    // 编译 OpenCL 程序
    err = clBuildProgram(g_program, 1, &g_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "OpenCL Program Build Log:\n" << build_log.data() << std::endl;
        exit(1);
    }

    // 创建内核
    g_kernel = clCreateKernel(g_program, "block_matching_multi", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Error creating kernel: " << err << std::endl;
        exit(1);
    }

    std::clog << "cl inited!" << std::endl;
    g_opencl_initialized = true;  // 标记为已经初始化
}

// 释放 OpenCL 资源
void releaseOpenCL() {
    if (!g_opencl_initialized) return;  // 如果没有初始化，则不释放

    clReleaseKernel(g_kernel);
    clReleaseProgram(g_program);
    clReleaseCommandQueue(g_queue);
    clReleaseContext(g_context);

    std::clog << "cl freed!" << std::endl;
    g_opencl_initialized = false;
}