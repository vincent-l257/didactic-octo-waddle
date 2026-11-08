// src/program/serve/async_report.hpp - a layer split's chunk report (the drafter's rows) off the last stage's path.
//
// The last stage's rows are copied to a spare buffer on its device (its own rows are free again at once), then to
// pinned host memory on a side stream while the stage reads its next chunk; a worker thread hands them to the report
// in chunk order.  drain() waits for every report given so far.  Without the buffers it reports in place.
#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace basalt::program::serve {

class AsyncReport {
public:
    using Fn = std::function<bool(const float* R_rows, int64_t T, int64_t p0, std::string& err)>;

    AsyncReport(Fn inner, int dev, int64_t row_floats) : inner_(std::move(inner)), dev_(dev), row_(row_floats) {
        worker_ = std::thread([this] { work(); });
    }
    AsyncReport(const AsyncReport&) = delete;
    AsyncReport& operator=(const AsyncReport&) = delete;
    ~AsyncReport() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev_);
        if (spare_) cudaFree(spare_);
        for (float* h : host_)
            if (h) cudaFreeHost(h);
        for (cudaEvent_t e : ev_)
            if (e) cudaEventDestroy(e);
        if (cs_) cudaStreamDestroy(cs_);
        cudaSetDevice(cur);
    }

    bool chunk(const float* R, int64_t T, int64_t p0, std::string& err) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!fail_.empty()) { err = fail_; return false; }
        }
        const size_t bytes = (size_t) T * (size_t) row_ * sizeof(float);
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev_);
        const bool room = reserve(bytes);
        cudaSetDevice(cur);
        if (!room) {   // no buffers: in place, after the reports before it
            if (!drain(err)) return false;
            return inner_(R, T, p0, err);
        }
        const int slot = next_;
        next_ ^= 1;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return !busy_[slot] || !fail_.empty(); });
            if (!fail_.empty()) { err = fail_; return false; }
            busy_[slot] = true;
        }
        cudaSetDevice(dev_);
        bool ok = cudaMemcpyAsync(spare_, R, bytes, cudaMemcpyDeviceToDevice, cs_) == cudaSuccess &&
                  cudaStreamSynchronize(cs_) == cudaSuccess &&   // the stage's rows are free again
                  cudaMemcpyAsync(host_[slot], spare_, bytes, cudaMemcpyDeviceToHost, cs_) == cudaSuccess &&
                  cudaEventRecord(ev_[slot], cs_) == cudaSuccess;
        cudaSetDevice(cur);
        if (!ok) {
            err = std::string("chunk report: ") + cudaGetErrorString(cudaGetLastError());
            std::lock_guard<std::mutex> lk(mu_);
            busy_[slot] = false;
            return false;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_.push_back({slot, T, p0});
        }
        cv_.notify_all();
        return true;
    }

    bool drain(std::string& err) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (jobs_.empty() && !running_) || !fail_.empty(); });
        if (!fail_.empty()) {
            err = fail_;
            fail_.clear();
            jobs_.clear();
            busy_[0] = busy_[1] = false;
            return false;
        }
        return true;
    }

private:
    struct Job { int slot; int64_t T, p0; };

    bool reserve(size_t bytes) {
        if (bytes <= cap_) return true;
        if (!drain_quiet()) return false;
        if (spare_) cudaFree(spare_);
        for (float*& h : host_) {
            if (h) cudaFreeHost(h);
            h = nullptr;
        }
        spare_ = nullptr;
        cap_ = 0;
        if (!cs_ && cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) return none();
        for (cudaEvent_t& e : ev_)
            if (!e && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return none();
        if (cudaMalloc((void**) &spare_, bytes) != cudaSuccess) return none();
        for (float*& h : host_)
            if (cudaHostAlloc((void**) &h, bytes, cudaHostAllocPortable) != cudaSuccess) return none();
        cap_ = bytes;
        return true;
    }
    bool none() {
        cudaGetLastError();
        if (spare_) cudaFree(spare_);
        spare_ = nullptr;
        for (float*& h : host_) {
            if (h) cudaFreeHost(h);
            h = nullptr;
        }
        cap_ = 0;
        return false;
    }
    bool drain_quiet() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return (jobs_.empty() && !running_) || !fail_.empty(); });
        return fail_.empty();
    }

    void work() {
        for (;;) {
            Job j{};
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
                if (jobs_.empty()) return;
                j = jobs_.front();
                jobs_.pop_front();
                running_ = true;
            }
            std::string e;
            bool ok = cudaEventSynchronize(ev_[j.slot]) == cudaSuccess;
            if (!ok) e = std::string("chunk report: ") + cudaGetErrorString(cudaGetLastError());
            else ok = inner_(host_[j.slot], j.T, j.p0, e);
            {
                std::lock_guard<std::mutex> lk(mu_);
                running_ = false;
                busy_[j.slot] = false;
                if (!ok && fail_.empty()) fail_ = e.empty() ? "chunk report failed" : e;
            }
            cv_.notify_all();
        }
    }

    Fn inner_;
    int dev_;
    int64_t row_;
    cudaStream_t cs_ = nullptr;
    float* spare_ = nullptr;
    float* host_[2] = {nullptr, nullptr};
    cudaEvent_t ev_[2] = {nullptr, nullptr};
    size_t cap_ = 0;
    int next_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    bool busy_[2] = {false, false};
    bool running_ = false;
    bool stop_ = false;
    std::string fail_;
    std::thread worker_;
};

}  // namespace basalt::program::serve
