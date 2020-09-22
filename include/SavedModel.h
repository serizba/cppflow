#ifndef CPPFLOW_SAVEDMODEL_H
#define CPPFLOW_SAVEDMODEL_H

#include "Model.h"

// load SavedModel created by tf v2.x
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
// https://www.tensorflow.org/guide/saved_model
class SavedModel : public Model {
protected:
    // no such op?
    using Model::restore;
    using Model::save;

public:
    SavedModel(const std::string& model_dir, const char* const tag = "serve", const std::vector<uint8_t>& config_options = {});

    // init_op is called automatically, no need to call again
    void init() {};
};

#endif // CPPFLOW_SAVEDMODEL_H
