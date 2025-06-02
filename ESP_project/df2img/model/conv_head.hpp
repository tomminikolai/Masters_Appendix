#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace conv_head
{
    const dl::Filter<int8_t> *get_functional_1_1_conv2d_5_1_biasadd_filter();
    const dl::Bias<int8_t> *get_functional_1_1_conv2d_5_1_biasadd_bias();
    const dl::Filter<int8_t> *get_functional_1_1_conv2d_6_1_biasadd_filter();
    const dl::Bias<int8_t> *get_functional_1_1_conv2d_6_1_biasadd_bias();
    const dl::Filter<int8_t> *get_functional_1_1_conv2d_7_1_biasadd_filter();
    const dl::Bias<int8_t> *get_functional_1_1_conv2d_7_1_biasadd_bias();
    const dl::Filter<int8_t> *get_functional_1_1_conv2d_8_1_biasadd_filter();
    const dl::Bias<int8_t> *get_functional_1_1_conv2d_8_1_biasadd_bias();
    const dl::Filter<int8_t> *get_functional_1_1_conv2d_9_1_biasadd_filter();
    const dl::Bias<int8_t> *get_functional_1_1_conv2d_9_1_biasadd_bias();
}
