#include "../include/SavedModel.h"

SavedModel::SavedModel(const std::string& model_dir, const char* const tag, const std::vector<uint8_t>& config_options) : Model() {
    status = TF_NewStatus();
    graph = TF_NewGraph();

    // Create the session.
    TF_SessionOptions* session_options = TF_NewSessionOptions();

    if (!config_options.empty())
    {
        TF_SetConfig(session_options, static_cast<const void*>(config_options.data()), config_options.size(), this->status);
        this->status_check(true);
    }

    TF_Buffer* run_options = TF_NewBufferFromString("", 0);
    TF_Buffer* meta_graph = TF_NewBuffer();

    int tag_len = 1;
    session = TF_LoadSessionFromSavedModel(session_options, run_options,
                model_dir.c_str(), &tag, tag_len, graph, meta_graph, status);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteBuffer(run_options);
    TF_DeleteBuffer(meta_graph);

    // Check the status
    status_check(true);
}
