#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"
#include <memory>
//#include "eval_op.h"

using namespace std;
class evaluation
{
public:
    std::string keyStr;
    int key_expr_id;
    evaluation(const std::vector<expression> &exprs);
//    evaluation(const std::vector<expression> &exprs, eval_op_proto_map &proto_map);
    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    double &get_result1();
    tensor &get_result();


private:
    std::map<std::string, double> kwargs_;
//    std::map<std::string, tensor> kwargs_tensor;
    double result_;
    std::vector<expression> exps;

    std::map<int, tensor> variables_;
//    std::map<string, tensor> kwargs_;
//    std::vector<std::shared_ptr<eval_op>> ops_; // instead of exprs_
    void Input1(expression &i, std::map<int, tensor > &key_str_arr);
    void Const1(expression &i, std::map<int, tensor > &key_str_arr);
    void Add1(expression &i, std::map<int, tensor > &key_str_arr);
    void Sub1(expression &i, std::map<int, tensor > &key_str_arr);
    void Mul1(expression &i, std::map<int, tensor > &key_str_arr);
    void OneMatrix( double &temp1, int row_2, int col_2, vector<double> &temp2,vector<double> &result);
    void MultMatrix(int row_1, int col_1, vector<double> &temp1, int row_2, int col_2, vector<double> &temp2, vector<double> &result);
    void MultMatrix2(int row_1, int col_1, vector<double> &temp1, int row_2, int col_2, vector<double> &temp2, vector<double> &result);

    void ReLU(expression &i, std::map<int, tensor> &key_str_arr);
    void Flatten(expression &i, std::map<int, tensor> &key_str_arr);
    void Input2d(expression &i, std::map<int, tensor> &key_str_arr);
    void Linear(expression &i, std::map<int, tensor> &key_str_arr);
    void MaxPool2d(expression &i, std::map<int, tensor> &key_str_arr);
    void Conv2d(expression &i, std::map<int, tensor> &key_str_arr);

    void Input(expression &i, std::map<string, double> &key_str, std::string &keys_, double &val_);


    void Const(expression &i, map<string, double> &key_str, string &keys_, double &val_);

    void Add(expression &i, map<string, double> &key_str, string &keys_, double &val_);

    void Sub(vector<int> &t_i, expression &i, map<string, double> &key_str, string &keys_, double &val_);

    void Mul(expression &i, map<string, double> &key_str, string &keys_, double &val_);



};


// class evaluation

#endif // EVALUATION_H
