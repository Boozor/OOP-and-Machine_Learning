//
// Created by 吉卓林 on 2020/10/18.
//

#include "eval_add.h"

void eval_add::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = value_;
    // retrieve a and b from variables
     // perform the computation with a and b
     // update variables
}

eval_add::eval_add(const expression &expr):
        eval_op(expr) {
}

tensor eval_add::compute(const tensor &a, const tensor &b) {
//    ... // make sure a and b to have the same shape
//    ... // create c to have the same shape as a and b
//    ... // add elements of a and b to obtain elements of c
//    return c;
    if(a.shape_.size() != b.shape_.size()){
        exception *e = new exception();
        printf("eval_add::compute throw exception\n");
        throw e;
    }
    vector<double> a1 = a.data_;
    vector<double> a2 = b.data_;
    for (int k = 0; k < a1.size(); k++) {
        a1[k] += a2[k];
    }
    tensor *c = new tensor();
    c->data_ = a1;
    printf("Add1 size:%lu\n", sizeof(c->data_));
}
