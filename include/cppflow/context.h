//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_CONTEXT_H
#define CPPFLOW2_CONTEXT_H

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

bool status_check(TF_Status* status) {

    if (TF_GetCode(status) != TF_OK) {
        throw std::runtime_error(TF_Message(status));
    }
    return true;
}

class context {

    public:
        static TFE_Context* get_context() {
            static context context; // Guaranteed to be destroyed.
            // Instantiated on first use.
            return context.tfe_context;
        }

        static TF_Status* get_status() {
            static context context; // Guaranteed to be destroyed.
            // Instantiated on first use.
            return context.tf_status;
        }
    private:
        context() {
            this->tf_status = TF_NewStatus();
            this->tfe_opts = TFE_NewContextOptions();
            this->tfe_context = TFE_NewContext(this->tfe_opts, tf_status);
            status_check(tf_status);
        }

        TF_Status* tf_status;
        TFE_Context* tfe_context;
        TFE_ContextOptions* tfe_opts;

    public:
        context(context const&)         = delete;
        void operator=(context const&)  = delete;

        ~context() {
            TFE_DeleteContextOptions(this->tfe_opts);
            TFE_DeleteContext(this->tfe_context);
        }
};

#endif //CPPFLOW2_CONTEXT_H
