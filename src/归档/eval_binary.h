//
// Created by 吉卓林 on 2020/10/18.
//

#include "../tensor.h"
#include "eval_op.h"

#ifndef MY449_EVAL_BINARY_H
#define MY449_EVAL_BINARY_H

#endif //MY449_EVAL_BINARY_H
class eval_binary: public eval_op {
    virtual tensor compute(const tensor &a, const tensor &b) = 0;
public:
    eval_binary(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs) final;
};