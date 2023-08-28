// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "atomic"
#include "chrono"
#include "condition_variable"
#include "functional"
#include "iostream"
#include "memory"
#include "mutex"
#include "thread"
#include "thread_pool.hpp"

namespace triton { namespace backend { namespace hps {
#ifndef _TIMER_H_
#define _TIMER_H_
class Timer {
 public:
  Timer() : _expired(true), _try_to_expire(false) {}

  Timer(const Timer& timer)
  {
    _expired = timer._expired.load();
    _try_to_expire = timer._try_to_expire.load();
  }

  ~Timer() { stop(); }

  void stop()
  {
    if (_expired) {
      return;
    }
    if (_try_to_expire) {
      return;
    }
    // wait for timer stop
    _try_to_expire = true;
    {
      std::unique_lock<std::mutex> locker(_mutex);
      _con_var_expired.wait(locker, [this] { return _expired = true; });
      if (_expired == true) {
        _try_to_expire = false;
      }
    }
  }

  void start(size_t interval, std::function<void()> task)
  {
    if (_expired == false) {
      return;
    }
    _expired = false;
    // launch thread
    //_thread = std::thread(std::bind(&Timer::run,this,task));
    std::thread([this, interval, task]() {
      while (!_try_to_expire) {
        std::this_thread::sleep_for(std::chrono::seconds(interval));
        task();
      }
      {
        std::lock_guard<std::mutex> locker(_mutex);
        _expired = true;
        _con_var_expired.notify_one();
      }
    }).detach();
  }

  void startonce(size_t delay, std::function<void()> task)
  {
    auto fn = [&, task, delay](size_t, size_t) {
      std::this_thread::sleep_for(std::chrono::seconds(delay));
      task();
    };
    ThreadPool::get().post(fn);
  }

 private:
  void run(size_t interval, std::function<void()> task)
  {
    while (!_try_to_expire) {
      std::this_thread::sleep_for(std::chrono::seconds(interval));
      task();
    }
    {
      std::lock_guard<std::mutex> locker(_mutex);
      _expired = true;
      _con_var_expired.notify_one();
    }
  }
  // The status of timer
  std::atomic<bool> _expired;
  std::atomic<bool> _try_to_expire;
  std::mutex _mutex;
  std::condition_variable _con_var_expired;
};
#endif

}}}  // namespace triton::backend::hps