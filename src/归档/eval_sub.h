//
// Created by 吉卓林 on 2020/10/24.
//

#ifndef MY449_EVAL_SUB_H
#define MY449_EVAL_SUB_H

#include "eval_op.h"

class eval_sub: public eval_op {
    tensor value_;
protected:
    eval_sub();
    eval_sub(const expression &expr) ;
public:
    tensor compute(const tensor &a, const tensor &b) override;
    void eval(vars_type &variables, const kwargs_type &kwargs);
//    static void store_prototype(std::__1::map<std::string, std::shared_ptr<eval_op>> map);
};


#endif //MY449_EVAL_SUB_H
