//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_CONTEXT_H
#define CPPFLOW2_CONTEXT_H

#include <memory>
#include <stdexcept>
#include <utility>

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

namespace cppflow {

    inline bool status_check(TF_Status* status) {
        if (TF_GetCode(status) != TF_OK) {
            throw std::runtime_error(TF_Message(status));
        }
        return true;
    }

    // Tensorflow only allows single context per process.
    // Setting different session options may cause undefined behaviors, see
    // https://github.com/tensorflow/tensorflow/issues/19083
    // https://github.com/tensorflow/tensorflow/issues/18861
    // Usage: Call cppflow::context::set_context_options(opts) if you need to set a session option.
    // Note that you also need to use the same session option when you load a model.
    class context {
        public:
            // Get global singleton context
            static const context& get_singleton();
            static TFE_Context* get_context();
            static TF_Status* get_status();

            // Set global context options, thread unsafe, must be called at the beginning.
            // Only the first call is valid. Later calls have no effect, but no error either.
            // Does NOT take over ownership of opts. Caller needs to deallocate opts.
            static void set_context_options(TFE_ContextOptions* opts);

        private:
            TFE_Context* tfe_context{nullptr};

            // Only used in set_context_options() and the constructor
            // Avoid passing in arguments while calling get_singleton()
            // Set to nullptr thereafter, no need to deallocate in deconstructor
            static TFE_ContextOptions* tfe_context_options;

            context();

        public:
            context(context const&) = delete;
            context& operator=(context const&) = delete;
            context(context&&) = delete;
            context& operator=(context&&) = delete;

            ~context();
    };
}

namespace cppflow {

    // C++17, move this to a cpp file and remove inline if compiled with C++14
    inline TFE_ContextOptions* context::tfe_context_options{nullptr};

    inline const context& context::get_singleton() {
        static context global_context;
        return global_context;
    }

    inline TFE_Context* context::get_context() {
        return get_singleton().tfe_context;
    }

    inline TF_Status* context::get_status() {
        thread_local std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> local_tf_status(TF_NewStatus(), &TF_DeleteStatus);
        return local_tf_status.get();
    }

    inline void context::set_context_options(TFE_ContextOptions* opts) {
        tfe_context_options = opts;
        context::get_singleton();
        tfe_context_options = nullptr;
    }

    inline context::context() {
        auto tf_status = context::get_status();
        if(tfe_context_options == nullptr) {
            tfe_context_options = TFE_NewContextOptions();
        }
        this->tfe_context = TFE_NewContext(tfe_context_options, tf_status);
        status_check(tf_status);
    }

    inline context::~context() {
        TFE_DeleteContext(this->tfe_context);
    }

}

#endif //CPPFLOW2_CONTEXT_H
