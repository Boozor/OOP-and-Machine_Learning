//
// Created by 吉卓林 on 2020/10/18.
//

#ifndef MY449_EVAL_CONST_H
#define MY449_EVAL_CONST_H

#include "eval_op.h"

class eval_const: public eval_op {
    tensor value_;
protected:
    eval_const() {}
    eval_const(const expression &expr);

public:
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map) {
//        assert(proto_map.find("Const") == proto_map.end());
        proto_map["Const"] = std::make_shared<eval_const>(); // where is expr?
    }
    void eval(vars_type &variables, const kwargs_type &kwargs)  override;


};
std::shared_ptr<eval_op> eval_const::clone(const expression &expr) {
    return std::make_shared<eval_const>(expr);
}

#endif //MY449_EVAL_CONST_H
