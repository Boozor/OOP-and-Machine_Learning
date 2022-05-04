//
// Created by 吉卓林 on 2020/10/18.
//

#ifndef MY449_EVAL_INPUT_H
#define MY449_EVAL_INPUT_H

#include "eval_op.h"
class eval_input: public eval_op {
    tensor value_;
protected:
    eval_input(){}
    eval_input(const expression &expr);
public:
    std::shared_ptr<eval_op> clone(const expression &expr) override;

    /*static void store_prototype(eval_op_proto_map &proto_map){
//    assert(proto_map.find("Const") == proto_map.end());
        proto_map["Input"] = std::make_shared<eval_input>(); // where is expr?
    }*/
    void eval(vars_type &variables, const kwargs_type &kwargs)  override;
};
/*std::shared_ptr<eval_op> eval_input::clone(const expression &expr) {
    return std::make_shared<eval_input>(expr);
}*/

#endif //MY449_EVAL_INPUT_H
