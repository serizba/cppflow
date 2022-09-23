// MIT License
//
// Copyright (c) 2020 liufeng27
// Copyright (c) 2022 Sergio Izquierdo
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*!
 *  @file       defer.h
 *  @author     liufeng27
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-11-07
 */
#pragma once

// C++ headers
#include <functional>

namespace cppflow {

class defer {
 public:
  typedef std::function<void ()> Func;

  explicit defer(const Func& func) : _func(func) {}
  ~defer() {
    _func();
  }

  defer(const defer&) = delete;
  defer(defer&&) = delete;
  defer& operator=(const defer&) = delete;
  void* operator new (size_t) = delete;
  void operator delete (void*) = delete;

 private:
    Func _func;
};  // Class defer

}  // namespace cppflow
