//
// Created by serizba on 27/6/20.
//

#include "cppflow/context.h"


namespace cppflow
{
	bool status_check(TF_Status* status)
	{
		if (TF_GetCode(status) != TF_OK)
		{
			throw std::runtime_error(TF_Message(status));
		}
		return true;
	}

	TFE_Context* context::get_context()
	{
		return global_context.tfe_context;
	}

	TF_Status* context::get_status()
	{
		thread_local std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> local_tf_status(
			TF_NewStatus(), &TF_DeleteStatus);
		return local_tf_status.get();
	}

	context::context(TFE_ContextOptions* opts)
	{
		auto tf_status = context::get_status();
		if (opts == nullptr)
		{
			std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
				TFE_NewContextOptions(), &TFE_DeleteContextOptions);
			this->tfe_context = TFE_NewContext(opts.get(), tf_status);
		}
		else
		{
			this->tfe_context = TFE_NewContext(opts, tf_status);
		}
		status_check(tf_status);
	}

	context::context(context && ctx)
	noexcept
	:
	tfe_context (std::exchange
	(ctx
	.
	tfe_context
	,
	nullptr
	)
	)
    {
    }

	context &context::operator=(context && ctx)
	noexcept
 {
        tfe_context = std::exchange(ctx.tfe_context, nullptr);
        return *this;
    }

	context::~context()
	{
		TFE_DeleteContext(this->tfe_context);
	}
}
