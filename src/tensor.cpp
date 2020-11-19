#include "cppflow/tensor.h"

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace cppflow
{
	tensor::tensor(enum TF_DataType type, const void* data, size_t len, const std::vector<int64_t>& shape)
	{
		this->tf_tensor = {TF_AllocateTensor(type, shape.data(), shape.size(), len), TF_DeleteTensor};
		memcpy(TF_TensorData(this->tf_tensor.get()), data, TF_TensorByteSize(this->tf_tensor.get()));
		this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
		status_check(context::get_status());
	}

	tensor::tensor(TFE_TensorHandle* handle)
	{
		this->tfe_handle = {handle, TFE_DeleteTensorHandle};
	}

	tensor::tensor(TF_Tensor* t)
	{
		this->tf_tensor = {t, TF_DeleteTensor};
		this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
		status_check(context::get_status());
	}

	tensor tensor::shape() const
	{
		auto op = TFE_NewOp(context::get_context(), "Shape", context::get_status());
		status_check(context::get_status());

		TFE_OpAddInput(op, this->tfe_handle.get(), context::get_status());
		status_check(context::get_status());

		// EXECUTE
		int n = 1;
		TFE_TensorHandle* res[1];
		TFE_Execute(op, res, &n, context::get_status());
		status_check(context::get_status());
		TFE_DeleteOp(op);

		tensor r;
		r.tf_tensor = {TFE_TensorHandleResolve(res[0], context::get_status()), TF_DeleteTensor};
		status_check(context::get_status());
		r.tfe_handle = {TFE_NewTensorHandle(r.tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
		status_check(context::get_status());

		return r;
	}

	std::string tensor::device(bool on_memory) const
	{
		std::string res;
		if (on_memory)
			res = TFE_TensorHandleBackingDeviceName(this->tfe_handle.get(), context::get_status());
		else
			res = std::string(TFE_TensorHandleDeviceName(this->tfe_handle.get(), context::get_status()));

		status_check(context::get_status());
		return res;
	}


	datatype tensor::dtype() const
	{
		return TFE_TensorHandleDataType(this->tfe_handle.get());
	}

#ifdef TENSORFLOW_C_TF_TSTRING_H_
	// For future version TensorFlow 2.4
	template<>
	tensor::tensor(const std::string& value) {
		TF_TString tstr[1];
		TF_TString_Init(&tstr[0]);
		TF_TString_Copy(&tstr[0], value.c_str(), value.size());

		*this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void*)tstr, sizeof(tstr), {});
	}
#else
	template <>
	tensor::tensor(const std::string& value)
	{
		size_t size = 8 + TF_StringEncodedSize(value.length());
		char* data = new char[value.size() + 8];
		for (int i = 0; i < 8; i++) { data[i] = 0; }
		TF_StringEncode(value.c_str(), value.size(), data + 8, size - 8, context::get_status());
		status_check(context::get_status());

		*this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void*)data, size, {});
		delete[] data;
	}

#endif // TENSORFLOW_C_TF_TSTRING_H_

	tensor::~tensor() = default;
}
