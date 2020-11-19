//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_CONTEXT_H
#define CPPFLOW2_CONTEXT_H

#include <memory>
#include <utility>
#include <stdexcept>

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

namespace cppflow {

    bool status_check(TF_Status* status);

    class context {
        public:
            static TFE_Context* get_context();
            static TF_Status* get_status();

        private:
            TFE_Context* tfe_context{nullptr};

        public:
            explicit context(TFE_ContextOptions* opts = nullptr);

            context(context const&) = delete;
            context& operator=(context const&) = delete;
            context(context&&) noexcept;
            context& operator=(context&&) noexcept;

            ~context();
    };

    // TODO: create ContextManager class if needed
    static const context global_context;

}

#endif //CPPFLOW2_CONTEXT_H
