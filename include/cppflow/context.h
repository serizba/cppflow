// MIT License
//
// Copyright (c) 2020 Sergio Izquierdo
// Copyright (c) 2020 Jiannan Liu
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
 *  @file       context.h
 *  @author     Jiannan Liu
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-06-27
 */

#ifndef INCLUDE_CPPFLOW_CONTEXT_H_
#define INCLUDE_CPPFLOW_CONTEXT_H_

// C headers
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

// C++ headers
#include <memory>
#include <stdexcept>
#include <utility>

namespace cppflow {

inline bool status_check(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    throw std::runtime_error(TF_Message(status));
  }
  return true;
}

class context {
 public:
  explicit context(TFE_ContextOptions* opts = nullptr);

  context(context const&) = delete;
  context(context&&) noexcept;

  ~context();

  context& operator=(context const&) = delete;
  context& operator=(context&&) noexcept;

  static TFE_Context* get_context();

  // only use get_status() for eager ops
  static TF_Status* get_status();

 private:
  TFE_Context* tfe_context{nullptr};
};  // Class context

// @todo create ContextManager class if needed
// Set new context, thread unsafe, must be called at the beginning.
// TFE_ContextOptions* tfe_opts = ...
// cppflow::get_global_context() = cppflow::context(tfe_opts);
inline context& get_global_context() {
    static context global_context;
    return global_context;
}
}  // namespace cppflow

namespace cppflow {

inline TFE_Context* context::get_context() {
  return get_global_context().tfe_context;
}

inline TF_Status* context::get_status() {
  thread_local std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>
      local_tf_status(TF_NewStatus(), &TF_DeleteStatus);
  return local_tf_status.get();
}

inline context::context(TFE_ContextOptions* opts) {
  auto tf_status = context::get_status();
  if (opts == nullptr) {
    std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)>
        new_opts(TFE_NewContextOptions(), &TFE_DeleteContextOptions);
    this->tfe_context = TFE_NewContext(new_opts.get(), tf_status);
  } else {
    this->tfe_context = TFE_NewContext(opts, tf_status);
  }
  status_check(tf_status);
}

inline context::context(context&& ctx) noexcept
    : tfe_context(std::exchange(ctx.tfe_context, nullptr)) {}

inline context& context::operator=(context&& ctx) noexcept {
  tfe_context = std::exchange(ctx.tfe_context, tfe_context);
  return *this;
}

inline context::~context() {
  TFE_DeleteContext(this->tfe_context);
}

}  // namespace cppflow

#endif  // INCLUDE_CPPFLOW_CONTEXT_H_
