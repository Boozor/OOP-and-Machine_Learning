#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>
#include <list>
#include "tensor.h"

class evaluation;

class expression
{
    friend class evaluation;
public:
    int exp_id;
    std::string opname;
    const char *opName;
    std::string optype;
    std::vector<int> input;
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);
    tensor get_op_param(const char *) const;

    string get_op_type() const;

}; // class expression

#endif // EXPRESSION_H
