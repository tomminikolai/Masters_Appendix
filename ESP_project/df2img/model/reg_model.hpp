#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_avg_pool2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_transpose.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_sigmoid.hpp"
#include "dl_layer_softmax.hpp"
#include "reg_head.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace reg_head;

class Reg_Head : public Model<int8_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
    // Declare layers as member variables
    //Transpose<int8_t> l0;
    //Transpose<int8_t> l10;
    Reshape<int8_t> l1;
    Conv2D<int8_t> l2;
    Conv2D<int8_t> l3;
    Conv2D<int8_t> l4;
    Conv2D<int8_t> l5;
    Conv2D<int8_t> l6;

    

public:
    Sigmoid<int8_t, float, QIFO> l7;

    /**
     * @brief Initialize layers in constructor function
     * 
     */
    Reg_Head() : //l0(Transpose<int8_t>({}, "l0")),
              l1(Reshape<int8_t>({1, 1, 6272}, "l11")),
              l2(Conv2D<int8_t>(-1, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_SAME_END, {}, 1, 1, "l12")),
              l3(Conv2D<int8_t>(-2, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), get_fused_gemm_1_activation(), PADDING_SAME_END, {}, 1, 1, "l13")),
              l4(Conv2D<int8_t>(-2, get_fused_gemm_2_filter(), get_fused_gemm_2_bias(), get_fused_gemm_2_activation(), PADDING_SAME_END, {}, 1, 1, "l14")),
              l5(Conv2D<int8_t>(-3, get_fused_gemm_3_filter(), get_fused_gemm_3_bias(), get_fused_gemm_3_activation(), PADDING_SAME_END, {}, 1, 1, "l15")),
              l6(Conv2D<int8_t>(-4, get_fused_gemm_4_filter(), get_fused_gemm_4_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l16")),
              l7(Sigmoid<int8_t, float, QIFO>(-8, "l17")){}
    /**
     * @brief call each layers' build(...) function in sequence
     * 
     * @param input 
     */
    void build(Tensor<int8_t> &input)
    {
        //this->l0.build(input);
        //this->l1.build(this->l0.get_output());
        this->l1.build(input);
        this->l2.build(this->l1.get_output());
        this->l3.build(this->l2.get_output());
        this->l4.build(this->l3.get_output());
        this->l5.build(this->l4.get_output());
        this->l6.build(this->l5.get_output());
        this->l7.build(this->l6.get_output());
    }

    /**
     * @brief call each layers' call(...) function in sequence
     * 
     * @param input 
     */
    void call(Tensor<int8_t> &input)
    {
        //this->l0.call(input);
        //input.free_element();

        //this->l1.call(this->l0.get_output());
        //this->l0.get_output().free_element();
        
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();
        
        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();
        
        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();
        
        //this->l10.call(this->l9.get_output());
        //this->l9.get_output().free_element();
    }
};