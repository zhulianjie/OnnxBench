
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, size_t size)
{
    int sum = 0;
    for (size_t i = 0; i < min(10000000ull, size); ++i)
    {
        sum += (threadIdx.x + 1) * a[i];
    }

    *c = sum;
}

__global__ void add2Kernel(int* c, const int* a, size_t size)
{
    int sum = 0;
    for (size_t i = 0; i < min(10000000ull, size); ++i)
    {
        sum += (threadIdx.x + 1) * a[i];
    }

    *c = sum;
}

dim3 grid{ 1000 };
dim3 block{ 320 };

void Sum()
{
    std::vector<int> left;
    auto size = 1024 * 1024 * 1024; // 1G
    left.resize(size);
    std::generate(left.begin(), left.end(), std::rand);

    int* dev_data = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    auto cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(0);
    }

    cudaMemcpy(dev_data, left.data(), left.size() * sizeof(left[0]), cudaMemcpyHostToDevice);
    int* dev_output = nullptr;

    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(0);
    }

    while (true)
    {
        auto beg = chrono::high_resolution_clock::now();
        addKernel << <grid, block >> > (dev_output, dev_data, size);

        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        std::cout << "Thread id " << std::this_thread::get_id()
            << " total time " << chrono::duration_cast<chrono::milliseconds>(end - beg).count() << "ms" << std::endl;

        int sumHost = 0;
        cudaMemcpy(&sumHost, dev_output, 4, cudaMemcpyDeviceToHost);

        std::cout << "sum is " << sumHost << std::endl;
    }
}

void Sum2()
{
    std::vector<int> left;
    auto size = 1024 * 1024 * 1024; // 1G
    left.resize(size);
    std::generate(left.begin(), left.end(), std::rand);

    int* dev_data = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    auto cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(0);
    }

    cudaMemcpy(dev_data, left.data(), left.size() * sizeof(left[0]), cudaMemcpyHostToDevice);
    int* dev_output = nullptr;

    cudaStatus = cudaMalloc((void**)&dev_output, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(0);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (true)
    {
        auto beg = chrono::high_resolution_clock::now();
        add2Kernel << <grid, block, 0, stream >> > (dev_output, dev_data, size);

        cudaStreamSynchronize(stream);
        auto end = chrono::high_resolution_clock::now();
        std::cout << "Thread id " << std::this_thread::get_id()
            << " total time " << chrono::duration_cast<chrono::milliseconds>(end - beg).count() << "ms" << std::endl;

        int sumHost = 0;
        cudaMemcpy(&sumHost, dev_output, 4, cudaMemcpyDeviceToHost);

        std::cout << "sum is " << sumHost << std::endl;
    }
}

__global__ void GetRandom(int* output)
{
    int result = 1;
    for (int i = 0; i < 100000; ++i)
    {
        result += threadIdx.x * i;
    }

    *output = result;
}

// Issue multiple GetRandom kernels.
void Thread1()
{
    std::vector<int> left;
    auto size = 1024 * 1024 * 1024; // 1G
    left.resize(size);
    std::generate(left.begin(), left.end(), std::rand);

    int result;

    //addWithCuda(&result, left.data(), left.data(), (uint32_t)left.size());

    cudaDeviceSynchronize();

}

// allocate 100M GPU memory and call memory set all the time.
void Thread2()
{
    int* dev_a = nullptr;
    size_t size = 1024 * 1024 * 100;    // 100MB
    auto result = cudaMalloc(&dev_a, size);
    if (result != cudaSuccess)
    {
        std::cout << "Malloc failed " << std::endl;
        exit(0);
    }

    while(true)
    {
        auto beg = chrono::high_resolution_clock::now();

        cudaMemset(dev_a, rand(), size * sizeof(*dev_a));

        auto end = chrono::high_resolution_clock::now();
        std::cout << "memory latency " << chrono::duration_cast<chrono::milliseconds>(end - beg).count() << "ms" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

#include <nvml.h>

void checkNvmlReturn(nvmlReturn_t result, const char* errorMessage) {
    if (result != NVML_SUCCESS) {
        std::cerr << errorMessage << ": " << nvmlErrorString(result) << std::endl;
        //exit(1);
    }
}

#define ToStr(idStr) \
case idStr : return #idStr;

#include <string>
std::string ID2Str(uint32_t id)
{
    switch (id)
    {
        ToStr(NVML_GPM_METRIC_GRAPHICS_UTIL)
        ToStr(NVML_GPM_METRIC_SM_UTIL)
        ToStr(NVML_GPM_METRIC_SM_OCCUPANCY)
        ToStr(NVML_GPM_METRIC_INTEGER_UTIL)
        ToStr(NVML_GPM_METRIC_ANY_TENSOR_UTIL)
        ToStr(NVML_GPM_METRIC_DFMA_TENSOR_UTIL)
        ToStr(NVML_GPM_METRIC_HMMA_TENSOR_UTIL)
        ToStr(NVML_GPM_METRIC_IMMA_TENSOR_UTIL)
        ToStr(NVML_GPM_METRIC_DRAM_BW_UTIL)
        ToStr(NVML_GPM_METRIC_FP64_UTIL)
        ToStr(NVML_GPM_METRIC_FP32_UTIL)
        ToStr(NVML_GPM_METRIC_FP16_UTIL)
        ToStr(NVML_GPM_METRIC_PCIE_TX_PER_SEC)
        ToStr(NVML_GPM_METRIC_PCIE_RX_PER_SEC)
    default:
        return "id:" + std::to_string(id);
    }
}

void PrintGPMMetrics()
{
    nvmlReturn_t result;
    result = nvmlInit();
    checkNvmlReturn(result, "Failed to initialize NVML");

    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    checkNvmlReturn(result, "Failed to get device count");

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    checkNvmlReturn(result, "Failed to get device handle");
    while (true)
    {
        nvmlGpmMetricsGet_t gpmMetrics;
        gpmMetrics.version = NVML_GPM_METRICS_GET_VERSION;

        nvmlGpmSample_t sample1, sample2;
        nvmlGpmSampleAlloc(&sample1);
        nvmlGpmSampleAlloc(&sample2);

        result = nvmlGpmSampleGet(device, sample1);
        checkNvmlReturn(result, "Failed to get first GPM sample");

        std::this_thread::sleep_for(std::chrono::seconds(1)); // Add a delay between samples

        result = nvmlGpmSampleGet(device, sample2);
        checkNvmlReturn(result, "Failed to get second GPM sample");

        gpmMetrics.sample1 = sample1;
        gpmMetrics.sample2 = sample2;

        for (uint32_t i = 1; i < gpmMetrics.numMetrics + 1; ++i)
        {
            gpmMetrics.metrics[i - 1].metricId = nvmlGpmMetricId_t(i);
        }

        gpmMetrics.metrics[13].metricId = NVML_GPM_METRIC_PCIE_TX_PER_SEC;
        gpmMetrics.metrics[14].metricId = NVML_GPM_METRIC_PCIE_RX_PER_SEC;
        gpmMetrics.numMetrics = 15;

        result = nvmlGpmMetricsGet(&gpmMetrics);
        checkNvmlReturn(result, "Failed to get GPM metrics");

        for (unsigned int j = 0; j < gpmMetrics.numMetrics; ++j) {
            std::cout << ID2Str(gpmMetrics.metrics[j].metricId)
                << ", Value: ";
            if (gpmMetrics.metrics[j].nvmlReturn)
                std::cout << "not avaiblable";
            else
                std::cout << gpmMetrics.metrics[j].value << std::endl;
            std::cout << endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    result = nvmlShutdown();
    checkNvmlReturn(result, "Failed to shutdown NVML");
}

int PrintStaus()
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t utilization;

    // Initialize NVML
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get the first GPU device
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result) {
        printf("Failed to get handle for device 0: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get the utilization rates
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (NVML_SUCCESS != result) {
        printf("Failed to get utilization rates: %s\n", nvmlErrorString(result));
        return 1;
    }

    printf("GPU Utilization: %u%%\n", utilization.gpu);
    printf("Memory Utilization: %u%%\n", utilization.memory);


    nvmlGpmSample_t t;

    nvmlFieldValue_t value[2];
    value[0].fieldId = NVML_GPM_METRIC_SM_UTIL;
    value[1].fieldId = NVML_GPM_METRIC_SM_OCCUPANCY;

    // Retrieve SM occupancy
    result = nvmlDeviceGetFieldValues(device, _countof(value), value);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get SM occupancy: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }

    std::cout << "SM util " << value[0].value.dVal << std::endl;;
    std::cout << "SM occupy " << value[1].value.dVal << std::endl;

    // Shutdown NVML
    nvmlShutdown();
    return 0;
}

void HelloCudaCPU();

int main()
{
    cudaEventDefault;
    
    HelloCudaCPU();

    /*std::thread(
        []()
        {
            while (true)
            {
                PrintGPMMetrics();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }).detach();*/

    /*std::thread(&Sum2).detach();

    Sum2();*/

    getchar();

    return 0;
}
