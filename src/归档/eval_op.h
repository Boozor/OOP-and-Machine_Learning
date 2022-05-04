#ifndef EVAP_OP_H
#define EVAL_OP_H

#include "tensor.h"
#include <map>
#include <vector>
#include "expression.h"


typedef std::map<int, tensor> vars_type;
typedef std::map<std::string, tensor> kwargs_type;


class eval_op{
protected:
    int expr_id_;
    std::string op_name_;
    std::vector<int> inputs_;
    std::map<std::string, double> op_params;
    std::map<std::string, tensor> op_params_tensor;
public:
    eval_op(const expression &expr);
    virtual ~eval_op();
    virtual tensor eval(vars_type &variables, const kwargs_type &kwargs) = 0;
};


class eval_const: public eval_op{
    tensor value_;
public:
    eval_const(const expression &expr);
    tensor eval(vars_type &variables , const kwargs_type &kwargs)override;

};

class eval_input: public eval_op{
public:
    eval_input(const expression &expr);
    tensor eval(vars_type &variables , const kwargs_type &kwargs)override;
};

class eval_add: public eval_op{
public:
    eval_add(const expression &expr);
    tensor eval(vars_type &variables , const kwargs_type &kwargs)override;
};

class eval_sub: public eval_op{
public:
    eval_sub(const expression &expr);
    tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};

class eval_mul: public eval_op{
public:
    eval_mul(const expression &expr);
    tensor eval(vars_type &variables, const kwargs_type &kwargs)override;
};

#endif

