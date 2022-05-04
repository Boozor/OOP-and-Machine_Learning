//
// Created by 吉卓林 on 2020/10/18.
//

#include "eval_const.h"
void eval_const::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = value_;
}

eval_const::eval_const(const expression &expr):
        eval_op(expr), value_(expr.get_op_param("value")) {
}












