//
// Created by 吉卓林 on 2020/10/18.
//

#include "eval_binary.h"
eval_binary::eval_binary(const expression &expr):
        eval_op(expr) {
}

void eval_binary::eval(vars_type &variables, const kwargs_type &kwargs) {
//    assert(inputs_.size() == 2);
    auto ita = variables.find(inputs_[0]);
    auto itb = variables.find(inputs_[1]);
//    ... // handle errors for ita and itb
    variables[expr_id_] = compute(ita->second, itb->second);
}