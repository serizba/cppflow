//
// Created by serizba on 27/6/20.
//
#pragma once
#ifndef CPPFLOW2_TENSOR_H
#define CPPFLOW2_TENSOR_H

#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <tensorflow/c/tf_tensor.h>
#include <tensorflow/c/eager/c_api.h>

#include "context.h"
#include "datatype.h"

namespace cppflow
{
	/**
	 * @class tensor
	 * @brief A TensorFlow eager tensor wrapper
	 *
	 */
	class tensor
	{
	public:
		tensor() = default;

		/**
		* Creates a tensor with the given values and specified shape
		* @tparam T A type that can be convertible into a tensor
		* @param values The values to be converted (in a flattened version)
		* @param shape The shape of the converted tensor
		*/
		template <typename T>
		tensor(const std::vector<T>& values, const std::vector<int64_t>& shape);

		/**
		* Creates a flat tensor with the given values
		* @tparam T A type that can be convertible into a tensor
		* @param values The values to be converted
		*/
		template <typename T>
		tensor(const std::initializer_list<T>& values);

		/**
		 * Creates a tensor with the given value
		 * @tparam T A type that can be convertible into a tensor
		 * @param value The value to be converted
		 */
		template <typename T>
		tensor(const T& value);

		/**
		 * @return Shape of the tensor
		 */
		tensor shape() const;

		/**
		 * @param on_memory If false, the function will return the name of the device that produced the tensor.
		 * If true, the function will return the name of the device in whose memory the tensor resides
		 * @return Returns the name of the device of the tensor
		 */
		std::string device(bool on_memory = false) const;


		/**
		 * @return The tensor datatype
		 */
		datatype dtype() const;

		/**
		 * Converts the tensor into a C++ vector
		 * @tparam T The c++ type (must be equivalent to the tensor type)
		 * @return A vector representing the flat tensor
		 */
		template <typename T>
		std::vector<T> get_data() const;


		~tensor();
		tensor(const tensor& tensor) = default;
		tensor(tensor&& tensor) = default;
		tensor& operator=(const tensor& other) = default;
		tensor& operator=(tensor&& other) = default;


		std::shared_ptr<TF_Tensor> tf_tensor;
		std::shared_ptr<TFE_TensorHandle> tfe_handle;

		explicit tensor(TFE_TensorHandle* handle);
		explicit tensor(TF_Tensor* t);

	private:

		tensor(enum TF_DataType type, const void* data, size_t len, const std::vector<int64_t>& shape);
	};
}


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace cppflow
{


	template <typename T>
	tensor::tensor(const std::vector<T>& values, const std::vector<int64_t>& shape) :
		tensor(deduce_tf_type<T>(), values.data(), values.size() * sizeof(T), shape)
	{
	}

	template <typename T>
	tensor::tensor(const std::initializer_list<T>& values) :
		tensor(std::vector<T>(values), {(int64_t)values.size()})
	{
	}

	template <typename T>
	tensor::tensor(const T& value) :
		tensor(std::vector<T>({value}), {})
	{
	}


	template <typename T>
	std::vector<T> tensor::get_data() const
	{
		auto res_tensor = TFE_TensorHandleResolve(this->tfe_handle.get(), context::get_status());
		status_check(context::get_status());

		// Check tensor data is not empty
		auto raw_data = TF_TensorData(res_tensor);
		//this->error_check(raw_data != nullptr, "Tensor data is empty");

		size_t size = TF_TensorByteSize(res_tensor) / TF_DataTypeSize(TF_TensorType(res_tensor));

		// Convert to correct type
		const auto T_data = static_cast<T*>(raw_data);
		return std::vector<T>(T_data, T_data + size);
	}

}

#endif //CPPFLOW2_TENSOR_H
