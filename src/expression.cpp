#include "expression.h"

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
    : input(inputs, inputs + num_inputs)
{
    exp_id = expr_id;
    opname = op_name;
    opName = op_name+expr_id;
    optype = op_type;
//    num_inputs = num_inputs;
}

void expression::add_op_param_double(
    const char *key,
    double value)
{
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
}

tensor expression::get_op_param(const char *string) const {
    return tensor();
}

string expression::get_op_type() const {
    return optype;
}
