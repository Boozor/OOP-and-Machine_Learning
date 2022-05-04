//
// Created by 吉卓林 on 2020/10/18.
//

#ifndef MY449_EVAL_CONST_H
#define MY449_EVAL_CONST_H

#include "eval_op.h"
#include "eval_binary.h"

class eval_add: public eval_binary {
    tensor value_;
protected:
    eval_add();
    eval_add(const expression &expr) ;
public:
    tensor compute(const tensor &a, const tensor &b) override;
    void eval(vars_type &variables, const kwargs_type &kwargs)  override;
    /*static void store_prototype(eval_op_proto_map &proto_map) {
//    assert(proto_map.find("Const") == proto_map.end());
        proto_map["Add"] = std::make_shared<eval_add>(); // where is expr?
    }*/
};


#endif //MY449_EVAL_CONST_H
