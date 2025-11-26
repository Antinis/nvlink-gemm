#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Naive GEMM kernel: C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void naive_gemm(const float* A, const float* B, float* C, 
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void print_usage(const char* prog_name) {
    printf("用法: %s -c <compute_gpu> -a <matrix_a_gpu> -b <matrix_b_gpu> -m <M> -n <N> -k <K>\n", prog_name);
    printf("参数说明:\n");
    printf("  -c: 执行计算的GPU ID (0-3)\n");
    printf("  -a: 矩阵A所在的GPU ID (0-3)\n");
    printf("  -b: 矩阵B所在的GPU ID (0-3)\n");
    printf("  -m: 矩阵维度M (矩阵A的行数)\n");
    printf("  -n: 矩阵维度N (矩阵B的列数)\n");
    printf("  -k: 矩阵维度K (矩阵A的列数/矩阵B的行数)\n");
    printf("示例: %s -c 0 -a 1 -b 2 -m 2048 -n 2048 -k 2048\n", prog_name);
}

int main(int argc, char** argv) {
    int compute_gpu = -1;
    int matrix_a_gpu = -1;
    int matrix_b_gpu = -1;
    int M = -1, N = -1, K = -1;
    
    // 解析命令行参数
    int opt;
    while ((opt = getopt(argc, argv, "c:a:b:m:n:k:h")) != -1) {
        switch (opt) {
            case 'c':
                compute_gpu = atoi(optarg);
                break;
            case 'a':
                matrix_a_gpu = atoi(optarg);
                break;
            case 'b':
                matrix_b_gpu = atoi(optarg);
                break;
            case 'm':
                M = atoi(optarg);
                break;
            case 'n':
                N = atoi(optarg);
                break;
            case 'k':
                K = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return 0;
        }
    }
    
    if (compute_gpu < 0 || matrix_a_gpu < 0 || matrix_b_gpu < 0 || 
        M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "错误: 缺少必要参数或参数无效\n\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("=== NVLink GEMM 性能测试 ===\n");
    printf("计算GPU: %d\n", compute_gpu);
    printf("矩阵A位置: GPU %d\n", matrix_a_gpu);
    printf("矩阵B位置: GPU %d\n", matrix_b_gpu);
    printf("矩阵维度: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    
    // 检查GPU数量
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("可用GPU数量: %d\n", device_count);
    
    if (compute_gpu >= device_count || matrix_a_gpu >= device_count || 
        matrix_b_gpu >= device_count) {
        fprintf(stderr, "错误: GPU ID超出范围\n");
        return EXIT_FAILURE;
    }
    
    // 检查NVLink连接
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, compute_gpu));
    printf("计算GPU: %s\n", prop.name);
    
    // 为IPC通信启用peer access
    CUDA_CHECK(cudaSetDevice(compute_gpu));
    if (matrix_a_gpu != compute_gpu) {
        int can_access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_gpu, matrix_a_gpu));
        if (!can_access) {
            fprintf(stderr, "警告: GPU %d 无法直接访问 GPU %d\n", compute_gpu, matrix_a_gpu);
        } else {
            cudaError_t err = cudaDeviceEnablePeerAccess(matrix_a_gpu, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            printf("已启用 GPU %d -> GPU %d 的 Peer Access\n", compute_gpu, matrix_a_gpu);
        }
    }
    
    if (matrix_b_gpu != compute_gpu && matrix_b_gpu != matrix_a_gpu) {
        int can_access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_gpu, matrix_b_gpu));
        if (!can_access) {
            fprintf(stderr, "警告: GPU %d 无法直接访问 GPU %d\n", compute_gpu, matrix_b_gpu);
        } else {
            cudaError_t err = cudaDeviceEnablePeerAccess(matrix_b_gpu, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            printf("已启用 GPU %d -> GPU %d 的 Peer Access\n", compute_gpu, matrix_b_gpu);
        }
    }
    
    // 分配矩阵A
    float *d_A = nullptr;
    CUDA_CHECK(cudaSetDevice(matrix_a_gpu));
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    
    // 初始化矩阵A
    std::vector<float> h_A(M * K);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    printf("矩阵A已分配在GPU %d上，大小: %.2f MB\n", matrix_a_gpu, 
           M * K * sizeof(float) / 1024.0 / 1024.0);
    
    // 分配矩阵B
    float *d_B = nullptr;
    CUDA_CHECK(cudaSetDevice(matrix_b_gpu));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    
    // 初始化矩阵B
    std::vector<float> h_B(K * N);
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    printf("矩阵B已分配在GPU %d上，大小: %.2f MB\n", matrix_b_gpu, 
           K * N * sizeof(float) / 1024.0 / 1024.0);
    
    // 分配矩阵C（结果矩阵，在计算GPU上）
    float *d_C = nullptr;
    CUDA_CHECK(cudaSetDevice(compute_gpu));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    printf("矩阵C已分配在GPU %d上，大小: %.2f MB\n", compute_gpu, 
           M * N * sizeof(float) / 1024.0 / 1024.0);
    
    // 配置kernel参数
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    printf("\n开始GEMM计算...\n");
    printf("Grid: (%d, %d), Block: (%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // 预热
    CUDA_CHECK(cudaSetDevice(compute_gpu));
    naive_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int num_iterations = 10;
    printf("执行 %d 次迭代...\n", num_iterations);
    
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < num_iterations; i++) {
        naive_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_time_ms = elapsed_ms / num_iterations;
    
    // 计算性能指标
    double flops = 2.0 * M * N * K; // 每次GEMM的浮点运算数
    double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;
    double bandwidth_gb = (M * K + K * N + M * N) * sizeof(float) / (avg_time_ms / 1000.0) / 1e9;
    
    printf("\n=== 性能结果 ===\n");
    printf("平均执行时间: %.3f ms\n", avg_time_ms);
    printf("性能: %.2f GFLOPS\n", gflops);
    printf("估计带宽: %.2f GB/s\n", bandwidth_gb);
    
    // 验证结果正确性（可选，只检查几个元素）
    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 简单验证：计算CPU上的一个元素来对比
    float cpu_result = 0.0f;
    for (int k = 0; k < K; k++) {
        cpu_result += h_A[k] * h_B[k * N];
    }
    printf("\n验证: C[0,0] = %.6f (GPU), %.6f (CPU验证)\n", h_C[0], cpu_result);
    printf("误差: %.6e\n", fabs(h_C[0] - cpu_result));
    
    // 清理资源
    CUDA_CHECK(cudaSetDevice(matrix_a_gpu));
    CUDA_CHECK(cudaFree(d_A));
    
    CUDA_CHECK(cudaSetDevice(matrix_b_gpu));
    CUDA_CHECK(cudaFree(d_B));
    
    CUDA_CHECK(cudaSetDevice(compute_gpu));
    CUDA_CHECK(cudaFree(d_C));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\n测试完成！\n");
    
    return 0;
}

