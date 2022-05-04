#ifndef PROGRAM_H
#define PROGRAM_H

#include "expression.h"

class evaluation;

class program
{
public:

    program();

    void append_expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int inputs[],
        int num_inputs);

    // return 0 for success
    int add_op_param_double(
        const char *key,
        double value);

    // return 0 for success
    int add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    evaluation *build();
private:
    int key_expr_id;
    std::string keyStr;
    std::vector<expression> vec;
}; // class program

#endif // PROGRAM_H
