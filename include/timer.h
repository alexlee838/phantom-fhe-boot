#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace Timer
{
    //
    // CPU timing data structures
    //
    static inline std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> cpuStartTimes;
    static inline std::unordered_map<std::string, double> cpuAccumulatedTimes; // in milliseconds

    //
    // GPU timing data structures
    //
#ifdef __CUDACC__
    struct GpuEventPair
    {
        cudaEvent_t start;
        cudaEvent_t stop;
    };
    static inline std::unordered_map<std::string, GpuEventPair> gpuEventPairs;
    static inline std::unordered_map<std::string, float> gpuAccumulatedTimes; // in milliseconds
#endif

    //
    // startCPUTimer
    //   Records the start time for a function identified by funcName.
    //
    inline void startCPUTimer(const std::string &funcName)
    {
        cpuStartTimes[funcName] = std::chrono::high_resolution_clock::now();
    }

    //
    // stopCPUTimer
    //   Records the end time, calculates the elapsed time, and adds it to the
    //   accumulated CPU time for funcName.
    //
    inline void stopCPUTimer(const std::string &funcName)
    {
        auto endTime = std::chrono::high_resolution_clock::now();

        // Look up the start time
        auto it = cpuStartTimes.find(funcName);
        if (it != cpuStartTimes.end())
        {
            auto startTime = it->second;
            double ms = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            cpuAccumulatedTimes[funcName] += ms;
            // Remove or leave it in if you need to time multiple intervals
            // cpuStartTimes.erase(it);
        }
        else
        {
            std::cerr << "[Timer] Warning: stopCPUTimer called without a matching startCPUTimer for \""
                      << funcName << "\".\n";
        }
    }

#ifdef __CUDACC__
    //
    // startGPUTimer
    //   Creates CUDA events and records the "start" event for funcName.
    //
    inline void startGPUTimer(const std::string &funcName)
    {
        GpuEventPair events;
        cudaEventCreate(&events.start);
        cudaEventCreate(&events.stop);

        // Record the start event
        cudaEventRecord(events.start, 0);

        gpuEventPairs[funcName] = events;
    }

    //
    // stopGPUTimer
    //   Records the "stop" event, measures elapsed time between
    //   the start and stop events, and adds to the accumulated GPU time.
    //
    inline void stopGPUTimer(const std::string &funcName)
    {
        auto it = gpuEventPairs.find(funcName);
        if (it != gpuEventPairs.end())
        {
            GpuEventPair &events = it->second;

            // Record the stop event
            cudaEventRecord(events.stop, 0);
            cudaEventSynchronize(events.stop);

            // Compute elapsed time (ms)
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, events.start, events.stop);
            gpuAccumulatedTimes[funcName] += ms;

            // Cleanup
            cudaEventDestroy(events.start);
            cudaEventDestroy(events.stop);

            // Remove or leave it if you want repeated intervals
            // gpuEventPairs.erase(it);
        }
        else
        {
            std::cerr << "[Timer] Warning: stopGPUTimer called without a matching startGPUTimer for \""
                      << funcName << "\".\n";
        }
    }
#else
    //
    // Dummy GPU functions if CUDA is not available.
    //
    inline void startGPUTimer(const std::string &funcName)
    {
        (void)funcName;
        std::cerr << "[Timer] Warning: startGPUTimer called, but CUDA is not enabled.\n";
    }

    inline void stopGPUTimer(const std::string &funcName)
    {
        (void)funcName;
        std::cerr << "[Timer] Warning: stopGPUTimer called, but CUDA is not enabled.\n";
    }
#endif

    //
    // printAccumulatedTimes
    //   Prints all accumulated CPU and GPU times to std::cout.
    //

    inline void printAccumulatedTimes()
    {
        // ===== CPU Timings =====
        // Move data from the unordered_map to a vector.
        std::vector<std::pair<std::string, double>> cpuTimes(cpuAccumulatedTimes.begin(), cpuAccumulatedTimes.end());
        // Sort in descending order by the timing value.
        std::sort(cpuTimes.begin(), cpuTimes.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.second > b.second; // largest first
                  });

        std::cout << "===== CPU Timings (ms) =====\n";
        for (const auto &kv : cpuTimes)
        {
            std::cout << kv.first << ": " << kv.second << " ms\n";
        }

#ifdef __CUDACC__
        // ===== GPU Timings =====
        // Move data from the unordered_map to a vector.
        std::vector<std::pair<std::string, float>> gpuTimes(gpuAccumulatedTimes.begin(), gpuAccumulatedTimes.end());
        // Sort in descending order by the timing value.
        std::sort(gpuTimes.begin(), gpuTimes.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.second > b.second; // largest first
                  });

        std::cout << "===== GPU Timings (ms) =====\n";
        for (const auto &kv : gpuTimes)
        {
            std::cout << kv.first << ": " << kv.second << " ms\n";
        }
#endif
    }

    //
    // clearAllTimings
    //   Resets all accumulated CPU and GPU timing data.
    //
    inline void clearAllTimings()
    {
        // Reset CPU timing totals
        for (auto &kv : cpuAccumulatedTimes)
            kv.second = 0.0;
    
    #ifdef __CUDACC__
        // Reset GPU timing totals
        for (auto &kv : gpuAccumulatedTimes)
            kv.second = 0.0f;
    #endif
    }
    

} // namespace Timer

#endif // TIMER_H
