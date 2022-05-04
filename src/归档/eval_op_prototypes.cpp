//
// Created by 吉卓林 on 2020/10/24.
//

#include "eval_op_prototypes.h"
#include "eval_op.h"
#include "eval_add.h"
//#include "eval_add.h"

eval_op_prototypes &eval_op_prototypes::instance() {
    static eval_op_prototypes instance; // the only instance
    return instance;
}

eval_op_prototypes::eval_op_prototypes() {
    eval_const::store_prototype(proto_map_);
    eval_input::store_prototype(proto_map_);
    eval_add::store_prototype(proto_map_);
    eval_sub::store_prototype(proto_map_);
    eval_mul::store_prototype(proto_map_);
//    ...
}