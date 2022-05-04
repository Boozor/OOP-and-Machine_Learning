//
// Created by 吉卓林 on 2020/10/24.
//

#include "eval_sub.h"

void eval_sub::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = value_;
     // retrieve a and b from variables
     // perform the computation with a and b
     // update variables
}
eval_sub::eval_sub(const expression &expr):
        eval_op(expr) {
}
tensor eval_sub::compute(const tensor &a, const tensor &b) {
    //TODO
}