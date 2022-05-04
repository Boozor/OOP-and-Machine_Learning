//
// Created by 吉卓林 on 2020/10/24.
//

#ifndef MY449_EVAL_OP_PROTOTYPES_H
#define MY449_EVAL_OP_PROTOTYPES_H

#include "eval_const.h"
#include "eval_input.h"
#include "eval_add.h"
#include "eval_sub.h"
#include "eval_mul.h"

class eval_op_prototypes {
// prevent creation of additional instances
    eval_op_prototypes(const eval_op_prototypes &) = delete;
    eval_op_prototypes();
    eval_op_proto_map proto_map_;
public:
    std::shared_ptr<eval_op> locate(std::string name);
    static eval_op_prototypes &instance(); // access the only instance
}; // class eval_op_prototypes
#endif //MY449_EVAL_OP_PROTOTYPES_H
