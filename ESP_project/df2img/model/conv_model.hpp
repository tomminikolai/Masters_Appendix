#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_avg_pool2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_transpose.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"
#include "conv_head.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace conv_head;

class Conv_Head : public Model<int8_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
    // Declare layers as member variables
    //Transpose<int8_t> l0;
    AvgPool2D<int8_t> l1;
    Conv2D<int8_t> l2;
    MaxPool2D<int8_t> l3;
    Conv2D<int8_t> l4;
    MaxPool2D<int8_t> l5;
    Conv2D<int8_t> l6;
    Conv2D<int8_t> l7;
    Conv2D<int8_t> l8;
    
    

    

public:
    MaxPool2D<int8_t> l9;

    /**
     * @brief Initialize layers in constructor function
     * 
     */
    Conv_Head() : //l0(Transpose<int8_t>({}, "l0")),
              l1(AvgPool2D<int8_t>(-7, {3, 3}, PADDING_VALID, {}, 3, 3, "l1")),
              l2(Conv2D<int8_t>(-7, get_functional_1_1_conv2d_5_1_biasadd_filter(), get_functional_1_1_conv2d_5_1_biasadd_bias(), NULL, PADDING_VALID, {}, 2, 2, "l2")),
              l3(MaxPool2D<int8_t>({3, 3}, PADDING_VALID, {}, 2, 2, "l3")),
              l4(Conv2D<int8_t>(-6, get_functional_1_1_conv2d_6_1_biasadd_filter(), get_functional_1_1_conv2d_6_1_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l4")),
              l5(MaxPool2D<int8_t>({3, 3}, PADDING_VALID, {}, 2, 2, "l5")),
              l6(Conv2D<int8_t>(-6, get_functional_1_1_conv2d_7_1_biasadd_filter(), get_functional_1_1_conv2d_7_1_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l6")),
              l7(Conv2D<int8_t>(-4, get_functional_1_1_conv2d_8_1_biasadd_filter(), get_functional_1_1_conv2d_8_1_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l7")),
              l8(Conv2D<int8_t>(-2, get_functional_1_1_conv2d_9_1_biasadd_filter(), get_functional_1_1_conv2d_9_1_biasadd_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l8")),
              l9(MaxPool2D<int8_t>({3, 3}, PADDING_VALID, {}, 1, 1, "l9")){}
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
        this->l8.build(this->l7.get_output());
        this->l9.build(this->l8.get_output());
        //this->l10.build(this->l9.get_output());
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
        
        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();
        
        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();
        
        //this->l10.call(this->l9.get_output());
        //this->l9.get_output().free_element();
        
    }
};