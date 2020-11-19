//
// Created by serizba on 31/7/20.
//


#include "cppflow/ops.h"

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {

    // Operators

    tensor operator+(const tensor& x, const tensor& y) {
        return add(x, y);
    }

    tensor operator-(const tensor& x, const tensor& y) {
        return sub(x, y);
    }

    tensor operator*(const tensor& x, const tensor& y) {
        return mul(x, y);
    }

    tensor operator/(const tensor& x, const tensor& y) {
        return div(x, y);
    }

    std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t) {
        std::string res =  to_string(t);
        return os << res;
    }


    std::string to_string(const tensor &t) {
        auto res_tensor = string_format({t.shape(), t}, "(tensor: shape=%s, data=\n%s)");
        auto res_tensor_h = TFE_TensorHandleResolve(res_tensor.tfe_handle.get(), context::get_status());
        status_check(context::get_status());

#ifdef TENSORFLOW_C_TF_TSTRING_H_
        // For future version TensorFlow 2.4
        //auto *t_str = reinterpret_cast<TF_TString *>(TF_TensorData(res_tensor_h));
        auto *t_str = (TF_TString *)(TF_TensorData(res_tensor_h));
        auto result = std::string(TF_TString_GetDataPointer(t_str), TF_TString_GetSize(t_str));
#else
        const char* dst[1] = {nullptr};
        size_t dst_len[1] = {3};
        TF_StringDecode(static_cast<char*>(TF_TensorData(res_tensor_h)) + 8, TF_TensorByteSize(res_tensor_h), dst, dst_len, context::get_status());
        status_check(context::get_status());
        auto result = std::string(dst[0], *dst_len);
#endif // TENSORFLOW_C_TF_TSTRING_H_

        TF_DeleteTensor(res_tensor_h);

        return result;
    }

}

