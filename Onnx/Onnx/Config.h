#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <optional>

namespace OnnxBenchmark
{
    struct TensorRTConfig
    {
        std::unordered_map<std::string, std::string> m_options;
    };

    enum InputType
    {
        Int32 = 0,
        FLOAT = 1
    };

    struct Config
    {
        std::string m_modelPath;
        std::vector<std::vector<std::int64_t>> m_inputShapes;
        std::uint64_t m_threadNum = 1;
        std::uint64_t m_iteration = 10;
        std::uint64_t m_batchNum = 100;
        std::uint64_t m_batchSize = 100;
        bool m_useSameSession = false;

        std::unordered_map<std::string, std::string> m_sessionOptions;

        TensorRTConfig m_tensorRT;
    };

}