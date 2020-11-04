#include <cmath>
#include <vector>
#include <cassert>

#include "cppflow/cppflow.h"

template<typename T>
void test(const std::initializer_list<T>& input) {
    std::vector<T> values = input;
    cppflow::tensor t(input);
    auto span_result = t.get_data<T>();

    assert(span_result.size() == values.size());
    for(size_t i = 0; i < span_result.size(); i++) {
        assert(std::abs(values[i]/span_result[i]-1.0f) < 1e-6);
    }
}

int main() {
    test({10, 20, 30});
    test({10.0, 20.1, 30.3});
    return 0;
}
