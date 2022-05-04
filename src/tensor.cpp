//
// Created by 吉卓林 on 2020/10/18.
//

#include "tensor.h"

tensor::tensor() : data_(1, 0) {
    //
//    data_.push_back(0);
} // scalar 0
tensor::tensor(double v) : data_(1, v) {
    //data_(1,v)
//    data_.push_back(v);
}// scalar v
extern "C"
tensor::tensor(int dim, size_t shape[], double data[])
/*:shape_(shape, shape+dim)*/ {
    shape_.assign(shape, shape + dim);
    int N = 1;
    double d = 0.000000;
// calculate N as shape[0]*shape[1]*...*shape[dim-1]
    for (int i = 0; i < dim; i++) {
        if (d == shape_[i]) {
            continue;
        }
        N *= shape_[i];
    }
    data_.assign(data, data + N);
}

size_t *tensor::get_shape_array() {
    return shape_.empty() ? nullptr : &shape_[0];
}

double *tensor::get_data_array() {
    printf("tensor::get_data_array size:%lu, data[0]:%f\n", data_.size(), data_[0]);
    return &data_[0];
}

int tensor::get_dim() const {
    printf("tensor::get_dim dim:%lu\n", shape_.size());
    return shape_.size();
}

// scalar only
double tensor::item() const {
//    assert(shape_.empty());
    return data_[0];
//    return 0.0;
}

double &tensor::item() {
//    assert(shape_.empty());
    return data_[0];
}

double tensor::at(size_t i) const {
//    assert(get_dim() == 1);
//    assert(i < shape_[0]);
    return data_[i];
}

double tensor::at(size_t i, size_t j) const {
//    assert(get_dim() == 2);
//    assert((i < shape_[0]) && (j < shape_[1]));
    return data_[i * shape_[1] + j];//i*col+j
}

/*tensor tensor::add(tensor t1,tensor t2){
    vector<double> a1 = t1.data_;//数组的第一个
    vector<double> a2 = t2.data_;
    for(int i= 0;i<t1.get_dim();i++){
        a1[i]+=a2[i];
    }
    return t1;
}
tensor tensor::sub(tensor t1,tensor t2){
    double *a1 = t1.get_data_array();
    return t1;
}
tensor tensor::mul(tensor t1,tensor t2){
    return t1;
}*/
/*tensor::tensor(tensor **pTensor) {

}*/
