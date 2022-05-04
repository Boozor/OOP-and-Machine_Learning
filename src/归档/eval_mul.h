//
// Created by 吉卓林 on 2020/10/24.
//

#ifndef MY449_EVAL_MUL_H
#define MY449_EVAL_MUL_H

#include "../tensor.h"
#include "eval_op.h"

class eval_mul: public eval_op {
    tensor value_;
public:
    eval_mul(const expression &expr);
    void eval(vars_type &variables, const kwargs_type &kwargs)  override;
};
#endif //MY449_EVAL_MUL_H
