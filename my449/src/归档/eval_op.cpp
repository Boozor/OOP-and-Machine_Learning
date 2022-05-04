#include "eval_op.h"

eval_op::~eval_op(){}

//const
eval_const::eval_const(const expression &expr):
        eval_op(expr), value_(expr.get_op_param("value")){}

eval_op::eval_op(expression const&){}
tensor eval_const::eval(vars_type &variables, const kwargs_type &kwargs ){
    variables[expr_id_] = value_;
}

//const

//input
eval_input::eval_input(const expression &expr):
        eval_op(expr){}
tensor eval_input::eval(vars_type &variables, const kwargs_type &kwargs ){
    variables[expr_id_];// = kwargs[];
}
//input

//add
eval_add::eval_add(const expression &expr):
        eval_op(expr){}
tensor eval_add::eval(vars_type &variables, const kwargs_type &kwargs ){
    //
    //tensor v = variables[inputs_[0]] + variables[inputs_[1]];
}
//add

//sub
eval_sub::eval_sub(const expression &expr):
        eval_op(expr){}
tensor eval_sub::eval(vars_type &variables, const kwargs_type &kwargs ){

}

//sub

//mul
eval_mul::eval_mul(const expression &expr):
        eval_op(expr){}
tensor eval_mul::eval(vars_type &variables, const kwargs_type &kwargs ){

}

//mul
