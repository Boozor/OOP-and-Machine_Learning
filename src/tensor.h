//
// Created by 吉卓林 on 2020/10/18.
//

#ifndef MY449_TENSOR_H
#define MY449_TENSOR_H

#include <cstdio>
#include <vector>

using namespace std;
class tensor {
public:
    tensor(); // scalar 0
    explicit tensor(double v); // scalar v
    tensor(int dim, size_t shape[], double data[]); // from C
//    tensor(tensor **pTensor);

    int get_dim() const;
// scalar only
    double item() const;
    double &item();
    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    size_t *get_shape_array();
    double *get_data_array();
    tensor add(tensor t1,tensor t2);
    tensor sub(tensor t1,tensor t2);
    tensor mul(tensor t1,tensor t2);
    std::vector<size_t> shape_;
    std::vector<double> data_;
private:
//    int dim_;
//    std::vector<size_t> shape_;
//    std::vector<double> data_;
};
#endif //MY449_TENSOR_H
