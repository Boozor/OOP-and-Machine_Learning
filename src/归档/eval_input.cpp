//
// Created by 吉卓林 on 2020/10/18.
//

#include "eval_input.h"
void eval_input::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = value_;
}

eval_input::eval_input(const expression &expr):
        eval_op(expr), value_(expr.get_op_param(expr.opName)) {
}

