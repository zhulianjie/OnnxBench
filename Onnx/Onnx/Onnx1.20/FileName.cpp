#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <numeric>
#include <onnxruntime_session_options_config_keys.h>
#include <random>
#include <memory>

using namespace std::chrono;

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

void CheckONNXErrorAndExit(const OrtApi& ort, OrtStatus* onnxStatus)
{
    if (onnxStatus != nullptr)
    {
        std::ostringstream os;
        os << "ONNX runtime returned error '" << ort.GetErrorMessage(onnxStatus) << "', code "
            << ort.GetErrorCode(onnxStatus);
        ort.ReleaseStatus(onnxStatus);
        std::cout << os.str() << std::endl;
        exit(1);
    }
}

#include <provider_options.h>

std::string print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<std::int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}

#include "Config.h"
using namespace OnnxBenchmark;

std::string GetDataTypeStr(ONNXTensorElementDataType type)
{
    std::string strType;
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "UInt32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "FLOAT";
    default:
        std::cout << "Usupported type " << type << std::endl;
        exit(0);
    }
}

static void PrintInputDim(Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& input_names,
    std::vector<ONNXTensorElementDataType>& dataTypes)
{
    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        dataTypes.emplace_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());

        std::cout << "\t" << input_names.at(i) << " : " << print_shape(shape) << std::endl;
        std::cout << "Data type " << GetDataTypeStr(dataTypes.back()) << std::endl;
    }

}


static void PrintOutputDim(Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& output_names)
{
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;

        auto type = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        std::cout << "Data type " << GetDataTypeStr(type) << std::endl;
    }
}

static void PrepareInput(const Config& config, std::vector<std::vector<std::vector<std::uint32_t>>>& tensors)
{
    std::cout << "Prepare tensors..." << std::endl;

    for (auto& shape : config.m_inputShapes)
    {
        std::cout << print_shape(shape) << std::endl;
    }

    std::vector<std::uint64_t> input_sizes;
    std::transform(config.m_inputShapes.begin(), config.m_inputShapes.end(),
        std::back_inserter(input_sizes),
        [](const auto& shape)
        {
            return std::accumulate(shape.begin(), shape.end(), 1,
                std::multiplies<std::uint64_t>());
        });

    tensors.resize(config.m_batchNum);
    for (auto& tensor : tensors)
    {
        for (auto& size : input_sizes)
        {
            std::vector<std::uint32_t> vec;
            vec.resize(size);

            std::generate(vec.begin(), vec.end(), []()
                {
                    return rand();
                });

            tensor.emplace_back(vec);
        }
    }
}

struct Metrics
{
    std::uint64_t m_batchlatency = 0;
};

static void RunInference(const Config& config,
    Ort::Session& session,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<std::vector<std::uint32_t>>>& inputsVector,
    const std::vector<ONNXTensorElementDataType>& types,
    Metrics& metrics)
{
    std::vector<const char*> input_names_char;
    std::transform(input_names.begin(), input_names.end(),
        std::back_inserter(input_names_char),
        [](const auto& item)
        {
            return item.c_str();
        });

    std::vector<const char*> output_names_char;
    std::transform(output_names.begin(), output_names.end(),
        std::back_inserter(output_names_char),
        [](const auto& item)
        {
            return item.c_str();
        });

    //Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, 0, OrtMemType::OrtMemTypeDefault);

    /*std::vector<std::vector<std::pair<uint32_t*, size_t>>> inputsVectorGPU;
    inputsVectorGPU.resize(inputsVectorCPU.size());
    for (size_t i = 0; i < inputsVectorCPU.size(); ++i)
    {
        auto& input = inputsVectorCPU[i];
        auto& inputGPU = inputsVectorGPU[i];
        inputGPU.resize(input.size());
        for (size_t j = 0; j < input.size(); ++j)
        {
            auto memory = cudaMa
        }
    }*/



    for (std::uint64_t iteration = 0; iteration < config.m_iteration; ++iteration)
    {
        auto beg = high_resolution_clock::now();
        for (auto& inputData : inputsVector)
        {
            try
            {
                auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::IoBinding ioBinding{ session };

                for (int j = 0; j < config.m_inputShapes.size(); ++j)
                {
                    auto type = types[j];
                    Ort::Value tensor = (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                        ? Ort::Value::CreateTensor<float>(memInfo,
                            (float*)inputData[j].data(),
                            inputData[j].size(),
                            config.m_inputShapes[j].data(),
                            config.m_inputShapes[j].size())
                        : Ort::Value::CreateTensor<int32_t>(memInfo,
                            (int32_t*)inputData[j].data(),
                            inputData[j].size(),
                            config.m_inputShapes[j].data(),
                            config.m_inputShapes[j].size());
                    ioBinding.BindInput(input_names[j].c_str(), tensor);
                }
                auto memInfo2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                for (int j = 0; j < session.GetOutputCount(); ++j)
                {
                    ioBinding.BindOutput(output_names[j].c_str(), memInfo2);
                }

                session.Run(Ort::RunOptions{ nullptr }, ioBinding);

                /*auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                    input_names_char.data(), inputData.data(), input_names_char.size(),
                    output_names_char.data(), output_names_char.size());*/

                    /*std::cout << "Output number " << output_tensors.size() << std::endl;
                    for (auto& output : output_tensors)
                    {
                        std::cout << print_shape(output.GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
                    }*/
            }
            catch (const Ort::Exception& exception) {
                std::cout << "ERROR running model inference: " << exception.what() << std::endl;
                exit(-1);
            }
        }
        auto end = high_resolution_clock::now();

        auto totalInUs = duration_cast<microseconds>(end - beg).count();
        metrics.m_batchlatency = totalInUs / inputsVector.size();

        /*std::cout << "thread Id " << std::this_thread::get_id() << "\t"
            << "total time " << totalInUs << " us; "
            << "Single " << totalInUs / inputsVector.size() << " us " << std::endl;*/
    }
}

#include <cuda_runtime.h>

static void RunInferenceNoCopy(const Config& config,
    Ort::Session& session,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<std::vector<std::uint32_t>>>& inputsVector,
    const std::vector<ONNXTensorElementDataType>& types,
    Metrics& metrics)
{
    std::vector<const char*> input_names_char;
    std::transform(input_names.begin(), input_names.end(),
        std::back_inserter(input_names_char),
        [](const auto& item)
        {
            return item.c_str();
        });

    std::vector<const char*> output_names_char;
    std::transform(output_names.begin(), output_names.end(),
        std::back_inserter(output_names_char),
        [](const auto& item)
        {
            return item.c_str();
        });

    Ort::MemoryInfo memory_info_cuda("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Allocator gpu_allocator(session, memory_info_cuda);
    std::vector<std::vector<Ort::Value>> gpuInputs;
    gpuInputs.resize(inputsVector.size());
    for(size_t inputIndex = 0; inputIndex < inputsVector.size(); ++inputIndex)
    {
        auto& inputData = inputsVector[inputIndex];
        auto& inputGPU = gpuInputs[inputIndex];
        for (int j = 0; j < config.m_inputShapes.size(); ++j)
        {
            auto size = inputData[j].size() * sizeof(uint32_t);
            void* cudaMemory = nullptr;
            cudaMalloc(&cudaMemory, size);
            cudaMemcpy(cudaMemory, inputData[j].data(), size, cudaMemcpyHostToDevice);

            auto type = types[j];
            Ort::Value tensor = (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                ? Ort::Value::CreateTensor(memory_info_cuda,
                    cudaMemory,
                    size,
                    config.m_inputShapes[j].data(),
                    config.m_inputShapes[j].size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                : Ort::Value::CreateTensor(memory_info_cuda,
                    cudaMemory,
                    size,
                    config.m_inputShapes[j].data(),
                    config.m_inputShapes[j].size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            inputGPU.emplace_back(std::move(tensor));
        }
    }

    auto batchSize = std::to_string(config.m_batchSize);
    for (std::uint64_t iteration = 0; iteration < config.m_iteration; ++iteration)
    {
        auto beg = high_resolution_clock::now();
        for (auto& inputData : gpuInputs)
        {
            try
            {
                auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::IoBinding ioBinding{ session };

                for (int j = 0; j < config.m_inputShapes.size(); ++j)
                {
                    ioBinding.BindInput(input_names[j].c_str(), inputData[j]);
                }
                Ort::MemoryInfo output_mem_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);                for (int j = 0; j < session.GetOutputCount(); ++j)
                {
                    ioBinding.BindOutput(output_names[j].c_str(), output_mem_info);
                }


                Ort::RunOptions run_options;/*
                run_options.SetRunLogVerbosityLevel(1);
                run_options.SetRunTag(batchSize.c_str());
                run_options.AddConfigEntry("gpu_graph_id", batchSize.c_str());*/
                session.Run(run_options, ioBinding);

                /*auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                    input_names_char.data(), inputData.data(), input_names_char.size(),
                    output_names_char.data(), output_names_char.size());*/

                    /*std::cout << "Output number " << output_tensors.size() << std::endl;
                    for (auto& output : output_tensors)
                    {
                        std::cout << print_shape(output.GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
                    }*/
            }
            catch (const Ort::Exception& exception) {
                std::cout << "ERROR running model inference: " << exception.what() << std::endl;
                exit(-1);
            }
        }
        auto end = high_resolution_clock::now();

        auto totalInUs = duration_cast<microseconds>(end - beg).count();
        metrics.m_batchlatency = totalInUs / inputsVector.size();

        /*std::cout << "thread Id " << std::this_thread::get_id() << "\t"
            << "total time " << totalInUs << " us; "
            << "Single " << totalInUs / inputsVector.size() << " us " << std::endl;*/
    }
}

#include <mutex>
#include <shared_mutex>
std::recursive_mutex g_mutex;

static void PrintTRTOptions(const std::unordered_map<std::string, std::string>& opts)
{
    std::lock_guard<std::mutex> guard(g_mutex);

    std::cout << "**********Thread Start" << std::this_thread::get_id() << std::endl;

    for (auto& pair : opts)
    {
        std::cout << "{" << pair.first << ":" << pair.second << "}" << std::endl;
    }

    std::cout << "**********Thread Done" << std::this_thread::get_id() << std::endl;
}

static void RunCapacity(Config config)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Onnx-c++-single");

    Ort::SessionOptions sessionOptions;

    for (auto& option : config.m_sessionOptions)
    {
        sessionOptions.AddConfigEntry(
            option.first.c_str(),
            option.second.c_str());
    }

    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    //sessionOptions.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    sessionOptions.SetIntraOpNumThreads(3);
    //sessionOptions.SetExecutionMode(ORT_PARALLEL);

    //sessionOptions.DisableCpuMemArena();

    const auto& api = Ort::GetApi();

    // tensor RT options.
    {
        OrtTensorRTProviderOptionsV2* trtOpt = nullptr;
        api.CreateTensorRTProviderOptions(&trtOpt);
        std::unique_ptr<OrtTensorRTProviderOptionsV2, 
            decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(
            trtOpt, api.ReleaseTensorRTProviderOptions);
        std::vector<const char*> keys;
        std::vector<const char*> values;

        std::transform(config.m_tensorRT.m_options.begin(),
            config.m_tensorRT.m_options.end(),
            std::back_inserter(keys),
            [](const auto& pair){ return pair.first.c_str(); });

        std::transform(config.m_tensorRT.m_options.begin(),
            config.m_tensorRT.m_options.end(),
            std::back_inserter(values),
            [](const auto& pair) { return pair.second.c_str(); });

        PrintTRTOptions(config.m_tensorRT.m_options);

        try
        {
            api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size());

            sessionOptions.AppendExecutionProvider_TensorRT_V2(*rel_trt_options);
        }
        catch (const Ort::Exception& exp)
        {
            std::cout << exp.what() << std::endl;
            exit(0);
        }
    }

    {
        /*OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        api.CreateCUDAProviderOptions(&cuda_options);
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
        std::vector<const char*> keys{ "enable_cuda_graph" };
        std::vector<const char*> values{ "1" };
        api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);

        sessionOptions.AppendExecutionProvider_CUDA_V2(*cuda_options);*/
    }


    OrtPrepackedWeightsContainer* container = nullptr;
    api.CreatePrepackedWeightsContainer(&container);

    auto size = config.m_modelPath.size() + 1;
    std::vector<wchar_t> wModelPath;
    wModelPath.resize(size);

    size_t charConverted = 0;
    mbstowcs_s(&charConverted, wModelPath.data(), size, config.m_modelPath.c_str(), size);

    std::cout << "*******" << config.m_modelPath << std::endl;
    std::cout << "*******" << "Total thread num " << config.m_threadNum << std::endl;

    std::cout << "***** create sessions***" << std::endl;
    std::vector<Ort::Session> sessions;
    for (std::uint64_t i = 0; i < config.m_threadNum; ++i)
    {
        sessions.emplace_back(env, wModelPath.data(), sessionOptions, container);

        if (config.m_useSameSession)
            break;
    }

    std::cout << "***** create sessions end***" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<ONNXTensorElementDataType> dataTypes;

    PrintInputDim(sessions[0], allocator, input_names, dataTypes);
    
    std::vector<std::string> output_names;
    PrintOutputDim(sessions[0], allocator, output_names);

    std::vector<std::vector<std::vector<std::uint32_t>>> tensors;
    PrepareInput(config, tensors);

    {
        std::cout << "******Warmup start ******" << std::endl;
        auto configWarmup = config;

        for (int i : {config.m_batchSize})
        {
            configWarmup.m_iteration = 1;
            configWarmup.m_batchNum = 1;
            configWarmup.m_batchSize = i;
            for (auto& shape : configWarmup.m_inputShapes)
            {
                shape[0] = i;
            }

            Metrics m;

            std::vector<std::vector<std::vector<std::uint32_t>>> warmupTensors;
            PrepareInput(configWarmup, warmupTensors);
            for (auto& session : sessions)
            {
                RunInferenceNoCopy(configWarmup, session, input_names, output_names,
                    warmupTensors,
                    dataTypes,
                    m);
            }
        }        std::cout << "******Warmup end******" << std::endl;
    }

    std::vector<std::thread> tasks;
    
    std::vector<Metrics> metrics;
    metrics.resize(config.m_threadNum);
    std::atomic_bool stop = false;

    for (std::uint64_t i = 0; i < config.m_threadNum; ++i)
    {
        auto& session = config.m_useSameSession ? sessions[0] : sessions[i];
        auto& metric = metrics[i];

        std::thread task([&]()
            {
                RunInferenceNoCopy(config, session, input_names, output_names,
                    tensors,
                    dataTypes,
                    metric);
                stop = true;
            });

        tasks.push_back(std::move(task));
    }

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        if (stop)
        {
            std::cout << "all tasks end" << std::endl;

            for (auto& task : tasks)
            {
                task.join();
            }
            break;
        }

        // calculate throughput
        if (std::none_of(metrics.begin(), metrics.end(),
            [](const auto& m)
            {
                return m.m_batchlatency == 0;
            }))
        {
            std::uint64_t latency = 0;
            // calculate total throughput
            std::uint64_t count = std::accumulate(metrics.begin(), metrics.end(),
                0ull, 
                [&](std::uint64_t value, const auto& m)
                {
                    latency += m.m_batchlatency;
                    return value + 1000000 / m.m_batchlatency * config.m_batchSize;
                });

            {
                std::lock_guard<std::mutex> guard(g_mutex);
                std::cout << "thread " << std::this_thread::get_id() << std::endl;
                std::cout << "batch Latency " << latency / metrics.size() << std::endl;
                std::cout << "Document throughput " << count << std::endl;
            }
        }
    }
}

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

template<typename T>
static bool GetOptional(T& des,
    boost::property_tree::ptree& ptree,
    const std::string& path)
{
    auto val = ptree.get_optional<T>(path);
    if (val)
    {
        des = *val;
        std::cout << "get config " << path << "  " << des << std::endl;

        return true;
    }

    return false;
}

template<typename T>
static bool GetRequired(T& des,
    boost::property_tree::ptree& ptree,
    const std::string& path)
{
    auto val = ptree.get_optional<T>(path);
    if (val)
    {
        des = *val;
        std::cout << "get config " << path << "  " << des << std::endl;
        return true;
    }

    exit(0);

    return false;
}

template<typename T>
static bool GetOptionalAndLogInt(T& des,
    boost::property_tree::ptree& ptree,
    const std::string& path);

template<typename T>
static bool GetOptional(boost::optional<T>& des,
    boost::property_tree::ptree& ptree,
    const std::string& path)
{
    des = ptree.get_optional<T>(path);
    return des.is_initialized();
}

template<typename T>
static bool GetOptionalAndLogInt(boost::optional<T>& des,
    boost::property_tree::ptree& ptree,
    const std::string& path);

static std::vector<Config> ReadConfig(const char* configPath)
{
    std::vector<Config> configs;
    std::cout << "Config file : " << configPath << std::endl;

    try
    {
        boost::property_tree::ptree ptree;
        boost::property_tree::xml_parser::read_xml(configPath, ptree, boost::property_tree::xml_parser::no_comments);
    
        auto rootSection = ptree.get_child("Root");
        
        for (auto& modelConfig : rootSection)
        {
            Config config;

            auto configSection = &modelConfig.second;
            GetRequired(config.m_modelPath, *configSection, "ModelPath");
            GetOptional(config.m_threadNum, *configSection, "ThreadNum");
            GetOptional(config.m_batchSize, *configSection, "BatchSize");
            GetOptional(config.m_iteration, *configSection, "IterationNum");
            GetOptional(config.m_batchNum, *configSection, "BatchNum");
            GetOptional(config.m_useSameSession, *configSection, "UseOneSession");

            auto shapesSection = configSection->get_child("Shapes");
            for (auto& shape : shapesSection)
            {
                config.m_inputShapes.emplace_back();
                config.m_inputShapes.back().emplace_back(config.m_batchSize);
                for (auto& dim : shape.second)
                {
                    config.m_inputShapes.back().emplace_back(dim.second.get_value<std::int64_t>());
                }

                std::cout << "Get shape from config " << print_shape(config.m_inputShapes.back()) << std::endl;
            }

            auto tensorRTSection = configSection->get_child_optional("TensorRT");
            if (tensorRTSection)
            {
                for (auto& option : *tensorRTSection)
                {
                    config.m_tensorRT.m_options.emplace(
                        option.second.get<std::string>("Key"),
                        option.second.get<std::string>("Value")
                    );
                }
            }

            configs.emplace_back(config);
        }

    }
    catch (const std::exception& exp)
    {
        std::cout << "Pass global hashTable config failed for " <<  exp.what() << std::endl;
        exit(0);
    }

    return configs;
}

#include <stdlib.h>

int main(int argc, const char** argv)
{
    _putenv("CUDA_FORCE_FLUSH=0");

    if (argc < 2)
    {
        std::cout << "No config file provided " << std::endl;
        return 0;
    }

    auto* configPath = argv[1];
    auto configs = ReadConfig(configPath);

    std::vector<std::thread> threads;
    for (auto& config : configs)
    {
        threads.emplace_back(std::thread([&]()
            {RunCapacity(config);}));
    }

    getchar();
    return 0;
}