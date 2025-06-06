#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace reg_head
{
    const dl::Filter<int8_t> *get_fused_gemm_0_filter();
    const dl::Bias<int8_t> *get_fused_gemm_0_bias();
    const dl::Activation<int8_t> *get_fused_gemm_0_activation();
    const dl::Filter<int8_t> *get_fused_gemm_1_filter();
    const dl::Bias<int8_t> *get_fused_gemm_1_bias();
    const dl::Activation<int8_t> *get_fused_gemm_1_activation();
    const dl::Filter<int8_t> *get_fused_gemm_2_filter();
    const dl::Bias<int8_t> *get_fused_gemm_2_bias();
    const dl::Activation<int8_t> *get_fused_gemm_2_activation();
    const dl::Filter<int8_t> *get_fused_gemm_3_filter();
    const dl::Bias<int8_t> *get_fused_gemm_3_bias();
    const dl::Activation<int8_t> *get_fused_gemm_3_activation();
    const dl::Filter<int8_t> *get_fused_gemm_4_filter();
    const dl::Bias<int8_t> *get_fused_gemm_4_bias();
}
