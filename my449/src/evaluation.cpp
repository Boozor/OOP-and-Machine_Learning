#include <assert.h>
#include <iostream>
#include "evaluation.h"
//#include "eval_input.h"
//#include "eval_const.h"

std::map<std::string, double> contextMap;
std::map<string, tensor> contextArrMap;

evaluation::evaluation(const std::vector<expression> &exprs)
        : /*result_(0)*/exps(exprs) {
}

void evaluation::add_kwargs_double(
        const char *key,
        double value) {
    printf("eval->add_kwargs_double key=%s\n", key);
    keyStr = key;
    printf("eval->add_kwargs_double key1=%s\n", keyStr.data());
    contextMap[key] = value;
    printf("eval->add_kwargs_double %f\n", value);
    tensor *t = new tensor(value);
    contextArrMap[key] = *t;
}

void evaluation::add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]) {
    /*printf("evaluation.add_kwargs_ndarray  key %s, value %p dim %d (",
            key, data, dim);
    for (int i = 0; i != dim; ++i)
        printf("%zu,", shape[i]);
    printf(")\n");*/
    keyStr = key;
    tensor *t = new tensor(dim, shape, data);
    contextArrMap[key] = *t;
}

int evaluation::execute() {

    map<string, double> key_str;
    string keys_;
    map<int, tensor> key_str_arr;
    variables_.clear();
    for (auto &i : exps) {
        printf("+++execute++++exp_id:%d, \n", i.exp_id);
        if (0 != i.opname.compare("")) {
            if (0 == i.optype.compare("Input")) {
//                Input(i, key_str, keys_, val_);
                Input1(i, key_str_arr);
//                ops_.push_back(std::make_shared<eval_input>(i));
            } else if (0 == i.optype.compare("Input2d")) {
                Input2d(i, key_str_arr);
            } else if (0 == i.optype.compare("Linear")) {
                Linear(i, key_str_arr);
            } else if (0 == i.optype.compare("Conv2d")) {//
                Conv2d(i, key_str_arr);
            }
        } else if (0 == i.opname.compare("")) {
            if (0 == i.optype.compare("Const")) {
//                Const(i, key_str, keys_, val_);
                Const1(i, key_str_arr);
//                ops_.push_back(std::make_shared<eval_const>(i));
            } else if (0 == i.optype.compare("Add")) {
//                printf("======execute.Add======\n");
//                Add(i, key_str, keys_, val_);
                Add1(i, key_str_arr);
//                ops_.push_back(std::make_shared<eval_add>(i));
            } else if (0 == i.optype.compare("Sub")) {
//                printf("======execute.Sub======\n");
//                Sub(t_i, i, key_str, keys_, val_);
                Sub1(i, key_str_arr);
//                ops_.push_back(std::make_shared<eval_sub>(i));
            } else if (0 == i.optype.compare("Mul")) {
//                Mul(i, key_str, keys_, val_);
//                printf("======execute.Mul======\n");
                Mul1(i, key_str_arr);
//                ops_.push_back(std::make_shared<eval_mul>(i));
            } else if (0 == i.optype.compare("ReLU")) {
//                printf("======execute.relu======\n");
                ReLU(i, key_str_arr);
            } else if (0 == i.optype.compare("Flatten")) {
//                printf("======execute.Flatten======\n");
                Flatten(i, key_str_arr);
            } else if (0 == i.optype.compare("MaxPool2d")) {
                MaxPool2d(i, key_str_arr);
            }
        }
    }
    printf("execute expid:%d, first:%f\n", key_expr_id,key_str_arr[key_expr_id].data_[0]);
    variables_[key_expr_id] = key_str_arr[key_expr_id];
    key_str_arr.clear();
    contextArrMap.clear();
    return 0;
}

void evaluation::Conv2d(expression &expr, std::map<int, tensor> &key_str_arr) {
//    tensor t = contextArrMap["x"];
    string key = "x";
    tensor t = contextArrMap["x"];
    if(t.shape_.size() == 0){
        key = "images";
        t = contextArrMap["images"];
    }
    int N = t.shape_[0];//10
    int C = t.shape_[1];//12  4
    int H = t.shape_[2];//15  5
    int W = t.shape_[3];//3
//    int H = t.shape_[1];//12  4
//    int W = t.shape_[2];//15  5
//    int C = t.shape_[3];//3
    tensor weight_t = contextArrMap["weight" + to_string(expr.exp_id)];
    tensor bias_t = contextArrMap["bias" + to_string(expr.exp_id)];
    tensor out_t = contextArrMap["out_channels" + to_string(expr.exp_id)];
    int out_channels = (int) out_t.data_[0];
    tensor kernel_ = contextArrMap["kernel_size" + to_string(expr.exp_id)];
    int kernel_size = (int) kernel_.data_[0];
//    int index_data = 0;
//    int index_filter = 0;
    double sum = 0;
    vector<double> result;
    size_t H_st = H + 1 - kernel_size;
    size_t W_st = W + 1 - kernel_size;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < out_channels; j++) {
            for (size_t h_it = 0; h_it < H_st; h_it++) {
                for (size_t w_it = 0; w_it < W_st; w_it++) {
                    for (size_t k = 0; k < C; k++) {
                        for (size_t m = 0; m < kernel_size; m++) {
                            for (size_t n = 0; n < kernel_size; n++) {
                                size_t data_idx = (i * C * H * W) + (k * H * W) + (n * W + h_it * W) + m + w_it;
                                size_t weight_idx =
                                        (j * C * kernel_size * kernel_size) + (k * kernel_size * kernel_size) +
                                        n * kernel_size + m;
                                sum +=   t.data_[data_idx] * weight_t.data_[weight_idx];
                            }
                        }
                    }
//                    count++;
                    result.push_back(sum + bias_t.data_[j]);
                    sum = 0;
                }
            }
        }
    }
    /*for (int n = 0; n < N; ++n) {
        for (int o = 0; o < out_channels; ++o) {
            for (int h = 0; h < H_stride; ++h) {//11
                for (int w = 0; w < W_stride; ++w) {//16
                    for (int c = 0; c < C; ++c) {
                        for (int n_ = 0; n_ < kernel_size; ++n_) {
                            for (int h_ = 0; h_ < kernel_size; ++h_) {
                                index_data = (n * C * H * W) + (c * H * W) + (h * H + h_ * W) + (w + n_);
                                index_filter = (o * C * kernel_size * kernel_size) + (c * kernel_size * kernel_size) +
                                               (h_ * kernel_size) + n_;
                                sum += t.data_[index_data] * weight_t.data_[index_filter];
                            }
                        }
                    }
                    result.push_back(sum + bias_t.data_[o]);
                    sum = 0;
                }
            }
        }
    }*/
    printf("new data out.size:%lu\n", result.size());
    t.data_ = result;
    vector<size_t> shape;
    shape.push_back(N);
    shape.push_back(out_channels);
    shape.push_back(H_st);
    shape.push_back(W_st);
    t.shape_ = shape;
    printf("Conv2d expid:%d\n", expr.exp_id);
    key_str_arr[expr.exp_id] = t;
    contextArrMap[key] = t;
}

/**
 * 如果输入的大小是(N,C,H,W)，那么输出的大小是(N,C,H_out,W_out)和池化窗口大小(kH,kW)的关系是：
$$out(N_i, C_j,k)=max^{kH-1}{m=0}max^{kW-1}{m=0} input(N_{i},C_j,stride[0]h+m,
 stride[1]w+n)$$
 * @param i
 * @param key_str_arr
 */
void evaluation::MaxPool2d(expression &expr, std::map<int, tensor> &key_str_arr) {
    printf("MaxPool2d opname:%s\n", expr.opname.c_str());
//    tensor t = contextArrMap["x"];
    string key = "x";
    tensor t = contextArrMap["x"];
    if(t.shape_.size() == 0){
        key = "images";
        t = contextArrMap["images"];
    }
    printf("t.data[0]:%f\n", t.data_[0]);
    int N = t.shape_[0];//10
    int C = t.shape_[1];//12  4 c  H
    int H = t.shape_[2];//15  5 h  W
    int W = t.shape_[3];//3     w  C

//    int H = t.shape_[1];//12  4 c  H
//    int W = t.shape_[2];//15  5 h  W
//    int C = t.shape_[3];//3     w  C
    tensor stride_t = contextArrMap["stride" + to_string(expr.exp_id)];
    int stride = (int) stride_t.data_[0];
    tensor kernel_t = contextArrMap["kernel_size" + to_string(expr.exp_id)];
    int kernel_size = (int) kernel_t.data_[0];
    printf("stride:%d, kernel_size:%d\n", stride, kernel_size);
    size_t H_r = H / kernel_size;
    size_t W_c = W / stride;
    printf("N:%d, C:%d, Ho:%d, Wo:%d\n", N, C, H_r, W_c);//4,5
//    对其中每个 kernel_size * kernel_size 的小块，取其中的最大值作为输出，输出的尺寸为 ( H / kernel_size, W / kernel_size)。
    vector<double> result;
    int rb = 0;//大行 12
    int cb = 0;//大列 15
    int ri = 0;//小行
    int ci = 0;//小列
    double max = 0;
    int index_tmp;
    size_t base;
    for (int n = 0; n < N; n++) {//10
        for (int c = 0; c < C; c++) {//3
            //从H*W的矩阵中取出每个3*3的矩阵中的最大值，然后进行返回；
            rb = 0;//大行 12
            cb = 0;//大列 15
            ri = 0;//小行i
            ci = 0;//小列j
            max = 0;
            base = (n * C + c) * H * W;
            for (int k = 0; k < H * W; k++) {
                index_tmp = base + (rb * kernel_size + ri) * W + (cb * stride + ci);
//                printf("max:%d,tmp:%d; ", max, t.data_[index_tmp]);
                if (t.data_[index_tmp] > max) {
                    max = t.data_[index_tmp];
                }
                ci++;
                if (ci >= stride) {
                    ri++;
                    ci = 0;
                }
                if (ri >= kernel_size) {
                    cb++;
                    ri = 0;
                    ci = 0;
//                    printf("%f,", max);
                    result.push_back(max);
                    max = 0;
                }
                if (cb >= W_c) {//5
                    rb++;
                    cb = 0;
                }
                if (rb >= H_r) {//4
                    rb = 0;//大行
                    cb = 0;//大列
                    ri = 0;//小行
                    ci = 0;//小列
                }
            }
        }
    }
    printf("\nnew data result.size:%lu, first:%f\n", result.size(), result[0]);
    t.data_ = result;
    vector<size_t> shape;
    shape.push_back(N);
    shape.push_back(C);
    shape.push_back(H/kernel_size);//4
    shape.push_back(W/stride);//5
    t.shape_ = shape;
    printf("MaxPool2d expid:%d\n", expr.exp_id);
    key_str_arr[expr.exp_id] = t;
    contextArrMap[key] = t;
}

/**
 * return input*weight + bias
 * w' 是将行转列
 * x * w' + b，b 从（10，）被 broadcast 成 （50，10）
 * @param i
 * @param key_str_arr
 */
void evaluation::Linear(expression &i, std::map<int, tensor> &key_str_arr) {
    printf("Linear ID：%d，opname:%s\n", i.exp_id, i.opname.c_str());
    tensor t = contextArrMap["x"];//50*100
    string key = "x";
    if(t.shape_.size() == 0){
        key = "images";
        t = contextArrMap[key];
    }
    printf("-1-data.size:%lu, shape.size:%lu\n", t.data_.size(), t.shape_.size());
    tensor w_t = contextArrMap["weight"+to_string(i.exp_id)];//10*100-》100*10
    printf("-2- size:%lu, expId:%d\n", w_t.data_.size(), i.exp_id);
    tensor b_t = contextArrMap["bias"+to_string(i.exp_id)];//10
    printf("-3-\n");
    int row = t.shape_[0];//50
    tensor t1 = contextArrMap["out_features"+to_string(i.exp_id)];
    printf("-4-\n");
    int col = int(t1.data_[0]);//10
    printf("-5- row:%d, col:%d\n", row,col);
//    tensor t2 = contextArrMap["in_features"];
//    int col1 = int(t1.data_[0]);//100
//    printf("Linear col:%d\n", col);

    //w-》w'
    vector<double> wv;
    printf("w' start ");
    for (int wr = 0; wr < w_t.shape_[1]; ++wr) {
        for (int wc = 0; wc < w_t.shape_[0]; ++wc) {
//            printf("%f,", w_t.data_[wr + w_t.shape_[1] * wc]);
            wv.push_back(w_t.data_[wr + w_t.shape_[1] * wc]);//i+j*col
        }
    }
    printf("w' end \n");

    printf("Linear w' size:%lu, first:%f\n", w_t.data_.size(), w_t.data_[0]);
    vector<double> result(row * col);//50*100 -> 50*10
    vector<double> tmp_v(row * col);
    //第一个矩阵的列数（column）和第二个矩阵的行数（row）相同才可以乘
    MultMatrix2(t.shape_[0], t.shape_[1], t.data_, w_t.shape_[1], w_t.shape_[0], wv, tmp_v);
//    MultMatrix(w_t.shape_[1], w_t.shape_[0], wv, t.shape_[0], t.shape_[1], t.data_,  tmp_v);
    printf("Linear w'*x size:%lu, first:%f\n", tmp_v.size(), tmp_v[0]);
    printf("Linear bias size:%lu, first:%f\n", b_t.data_.size(), b_t.data_[0]);
    vector<double> r;
    printf("r.row:%d, col:%d\n", row, col);
    for (int k = 0; k < row; ++k) {
        for (int m = 0; m < col; ++m) {
            r.push_back(b_t.data_[m] + tmp_v[k * col + m]);
        }
    }
    printf("AddMatrix after r.size:%lu, r[0]:%f\n", r.size(), r[0]);
    t.data_ = r;
    vector<size_t> shape;
    shape.push_back(row);
    shape.push_back(col);
    t.shape_ = shape;
    key_str_arr[i.exp_id] = t;
    contextArrMap[key] = t;
}

/**
 * 要求是把 NWHC 转换为 NCHW，并且这个转换不是 in-place 的，即应该开一块新的数组来存数据，不改动原数组里的数据。
实现时遍历一遍原数组，由于用的是一维数组，只要把高维数组和一维数组下标转换关系想好即可。
 例如二维情况下，二维数组 A[H][W] 中的元素 A[h][w] 如果展开成一维的数组 B[H*W] 中的元素就是 B[h*W+w]
 * @param expr
 * @param key_str_arr
 */
void evaluation::Input2d(expression &expr, std::map<int, tensor> &key_str_arr) {
    printf("Input2d start----expr.opname:%s\n", expr.opname.c_str());
    tensor t = contextArrMap[expr.opname];
    int N = t.shape_[0];//50
    int H = t.shape_[1];//10
    int W = t.shape_[2];//11
    int C = t.shape_[3];//3
    printf("N:%d, C:%d, H:%d, W:%d\n", N, C, H, W);//4,5
    vector<size_t> shape;
    shape.push_back(N);//50
    shape.push_back(C);//3
    shape.push_back(H);//10;
    shape.push_back(W);//11;
    t.shape_ = shape;
    vector<double> result(N * H * W * C);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            for (int k = 0; k < H; k++) {
                for (int m = 0; m < W; m++) {
                    result[m + W * (k + H * (j + C * i))] = t.data_[j + C * (m + W * (k + H * i))];
                }
            }
        }
    }
    printf("new data first:%f\n", result[0]);
    t.data_ = result;
    key_str_arr[expr.exp_id] = t;
    contextArrMap[expr.opname] = t;
//    contextArrMap[expr.opname+to_string(expr.exp_id)] = contextArrMap[expr.opname];
//    contextArrMap[expr.opname] = nullptr;
}

void evaluation::Flatten(expression &i, std::map<int, tensor> &key_str_arr) {
    printf("-------------Flatten \n");
    string key = "x";
    tensor t = contextArrMap["x"];
    if(t.shape_.size() == 0){
        key = "images";
        t = contextArrMap["images"];
    }
    vector<double> data = t.data_;
    printf("-------------Flatten.shape.length:%lu\n", t.shape_.size());
    int col = t.shape_[1];
    int row = t.shape_[0];
    int shape3 = t.shape_[2];
    int shape4 = t.shape_[3];
    printf("-------------Flatten.shape.row:%d,col:%d shape3:%d shape4:%d \n", row, col, shape3, shape4);
    vector<size_t> shape;
    shape.push_back(row);
    shape.push_back(col * shape3 * shape4);
    t.shape_ = shape;
    printf("-------------after Flatten.shape.length:%lu\n", shape.size());
    key_str_arr[i.exp_id] = t;
    contextArrMap[key] = t;
}

void evaluation::ReLU(expression &i, std::map<int, tensor> &key_str_arr) {
    string key = "x";
    tensor t = contextArrMap["x"];
    if(t.shape_.size() == 0){
        key = "images";
        t = contextArrMap["images"];
    }
    vector<double> data = t.data_;
    int length = data.size();
    printf("-------------ReLU.length:%d\n", length);
    for (int i = 0; i < length; ++i) {
        t.data_[i] = std::max((double) 0.0, data[i]);
    }
    key_str_arr[i.exp_id] = t;
    contextArrMap[key] = t;
}

void evaluation::Mul(expression &i, map<string, double> &key_str, string &keys_, double &val_) {
    val_ = 1.0;
    for (auto q : i.input) {
        keys_ = "t" + to_string(q);
        val_ *= key_str[keys_];
    }
    keys_ = "t" + to_string(i.exp_id);
    key_str[keys_] = val_;
//    key_op_type = 2;
}

void evaluation::Sub(vector<int> &t_i, expression &i, map<string, double> &key_str, string &keys_, double &val_) {
    val_ = 0.0;
    t_i = i.input;
    keys_ = "t" + to_string(t_i[0]);
    val_ = key_str[keys_];
    keys_ = "t" + to_string(t_i[1]);
    val_ = val_ - key_str[keys_];
    keys_ = "t" + to_string(i.exp_id);
    key_str[keys_] = val_;
//    key_op_type = 2;
}

void evaluation::Add(expression &i, map<string, double> &key_str, string &keys_, double &val_) {
    val_ = 0.0;
    for (auto q : i.input) {
        keys_ = "t" + to_string(q);
        val_ += key_str[keys_];
    }
    keys_ = "t" + to_string(i.exp_id);
    key_str[keys_] = val_;
//    key_op_type = 2;
}

void evaluation::Const(expression &i, map<string, double> &key_str, string &keys_, double &val_) {
    keys_ = "t" + to_string(i.exp_id);
    key_str[keys_] = contextMap["value"];
    val_ = key_str[keys_];
//    key_op_type = 0;
}

void evaluation::Input(expression &i, map<string, double> &key_str, string &keys_, double &val_) {
    keys_ = "t" + to_string(i.exp_id);
    key_str[keys_] = contextMap[i.opname];
    val_ = key_str[keys_];
//    key_op_type = 1;
}

void evaluation::Input1(expression &i, std::map<int, tensor> &key_str_arr) {
    key_str_arr[i.exp_id] = contextArrMap[i.opname];
//    kwargs_[i.opname] = contextArrMap[i.opname];
    printf("Input size:%lu\n", sizeof(contextArrMap[i.opname].shape_.size()));
}

void evaluation::Const1(expression &i, std::map<int, tensor> &key_str_arr) {

    try {
        key_str_arr[i.exp_id] = contextArrMap[to_string(i.exp_id)];
//    kwargs_["value"] = contextArrMap["value"];
        tensor t = key_str_arr[i.exp_id];
        double *d = t.get_data_array();
        printf("Const1 *d:%lf\n", *d);
        printf("Const1 size:%lu\n", sizeof(contextArrMap[i.opname]));
    } catch (exception &e) {
        std::cout << e.what() << std::endl;
    }
}

void evaluation::Add1(expression &i, std::map<int, tensor> &key_str_arr) {
    tensor t0;
    tensor t1;
    tensor t2;
    vector<double> a1;
    vector<double> a2;
    std::vector<int> input = i.input;
//    int j = 0;
    printf("Add1 input.size=%lu\n", input.size());
    printf("Add1 key_expr_id=%d \n", key_expr_id);//4
//    for (auto q: input) {
//        printf("Add1 exp_id =%d \n", q);//0,3
    /*if (j == 0) {
        key_str_arr[key_expr_id] = key_str_arr[q];//TODO  +=
        j++;
        continue;
    }*/
    printf("+++++++exp_id:%d\n", input[0]);
    printf("+++++++exp_id:%d\n", input[1]);
    t1 = key_str_arr[input[1]];
//        t2 = key_str_arr[q];
    t2 = key_str_arr[input[0]];

    a1 = t1.data_;
    a2 = t2.data_;
    for (int k = 0; k < a1.size(); k++) {
        a1[k] += a2[k];
    }
    t1.data_ = a1;
    key_str_arr[i.exp_id] = t1;
    key_str_arr[key_expr_id] = t1;
//    }
    printf("Add1 size:%lu\n", sizeof(key_str_arr[key_expr_id]));
}

void evaluation::Sub1(expression &i, std::map<int, tensor> &key_str_arr) {
    tensor t1;
    tensor t2;
    vector<double> a1;
    vector<double> a2;
    std::vector<int> input = i.input;
    printf("Sub1 input.size=%lu\n", input.size());
    printf("+++++++exp_id:%d\n", input[0]);
    printf("+++++++exp_id:%d\n", input[1]);
    t1 = key_str_arr[input[0]];
    t2 = key_str_arr[input[1]];
    a1 = t1.data_;
    a2 = t2.data_;
    for (int k = 0; k < a1.size(); k++) {
        a1[k] -= a2[k];
    }
    t1.data_ = a1;
    key_str_arr[i.exp_id] = t1;
    key_str_arr[key_expr_id] = t1;
}

void evaluation::Mul1(expression &i, std::map<int, tensor> &key_str_arr) {
    tensor t0;
    tensor t1;//结果
    tensor t2;//临时值
    vector<double> a1;
    vector<double> a2;
    vector<double> a3;
    vector<double> a0;
    int s3 = 1;
    double a = 1;
    int row = 0;
    int col = 0;
    int j = 0;
    std::vector<int> input = i.input;
//    printf("Mul1 input.size=%lu \n", input.size());
    printf("Mul1 key_expr_id=%d \n", key_expr_id);
    for (auto q:input) {
        printf("+++++++exp_id:%d\n", q);
        if (j == 0) {
            key_str_arr[i.exp_id] = key_str_arr[q];
            j++;
            continue;
        }
        t1 = key_str_arr[i.exp_id];
        t2 = key_str_arr[q];
        a1 = t1.data_;
//        printf("a1.size:%lu\n", a1.size());
        a2 = t2.data_;
//        printf("a2.size:%lu\n", a2.size());
        if (a1.size() == 1) {
//            printf("-a2--1: %f\n", a2[0]);
            a = a1[0];
            row = t2.shape_[0];
            col = t2.shape_[1];
            a0 = a2;
            s3 = t2.shape_[0] * t2.shape_[1];
        }
        if (a2.size() == 1) {
            a = a2[0];
            row = t1.shape_[0];
            col = t1.shape_[1];
            a0 = a1;
            s3 = t1.shape_[0] * t1.shape_[1];
//            printf("-a2--s3:%d\n", s3);
        }
        if (a1.size() > 1 && a2.size() > 1) {
            s3 = t1.shape_[0] * t2.shape_[1];
//            printf("-a3--s3:%d\n", s3);
        }
//        printf("+++++++s3:%d\n", s3);
        a3.assign(s3, 0);
        if (a1.size() > 1 && a2.size() > 1) {
            MultMatrix(t1.shape_[0], t1.shape_[1], a1, t2.shape_[0], t2.shape_[1], a2, a3);
        } else {
            if (a1.size() == 1) {
                OneMatrix(a, t2.shape_[0], t2.shape_[1], a2, a3);
            } else {
                OneMatrix(a, t1.shape_[0], t1.shape_[1], a1, a3);
            }
        }
//        printf("a3.size:%lu,\n", a3.size());
//        printf("-a3--1: %f\n", a3[0]);
        t1.data_ = a3;
        if (a1.size() > 1 && a2.size() > 1) {
            t1.shape_[1] = t2.shape_[1];//12= 13，
        } else if (a1.size() == 1) {
            t1.shape_.push_back(t2.shape_[0]);
            t1.shape_.push_back(t2.shape_[1]);
        } else if (a2.size() == 1) {
            printf("--continue--:%lu,\n", a3.size());
        }
        key_str_arr[i.exp_id] = t1;
        j++;
    }
    key_str_arr[key_expr_id] = t1;

    printf("Mul1.output tmp size:%lu\n", sizeof(key_str_arr[key_expr_id]));
}

void evaluation::OneMatrix(double &temp1, int row_2, int col_2, vector<double> &temp2, vector<double> &result) {
    int row_result, col_result;
    int num = 0;
    for (row_result = 0; row_result < row_2; row_result++) {//12
        for (col_result = 0; col_result < col_2; col_result++) {//13
            result[num] = 0;
            //对于m*n大小数组 如需访问其i*j元素 其对应的一维坐标为(i-1)*n+j
            //temp1 为row_1*col_1 需访问其col_row*(i+1)的元素 对应一维坐标为(col_result-1)*col_1+(i+1)
            //temp2 为row_2*col_2 需访问其(i+1)*col_result的元素 对应一维坐标为i*col_2+col_result
            //result[num] += temp1[(row_result - 1)*col_1 + i] * temp2[i*row_2 + col_result-(i+1)];
            result[num] = temp1 * temp2[row_result * col_2 + col_result];//1*12+0
            num++;
        }
    }
}

void evaluation::MultMatrix(int row_1, int col_1, vector<double> &temp1, int row_2, int col_2, vector<double> &temp2,
                            vector<double> &result) {
//    int times = col_1;//times=col_1=row_2 为确定result某元素时进行的乘法(加法)次数
    int row_result, col_result;
    int num = 0;
    //num为一维数组形式的result的索引
    //row_result col_result为二维数组形式的result的索引
    for (row_result = 0; row_result < row_1; row_result++) {
        for (col_result = 0; col_result < col_2; col_result++) {
            result[num] = 0;
            for (int i = 0; i < row_2; i++) {
                //对于m*n大小数组 如需访问其i*j元素 其对应的一维坐标为(i-1)*n+j
                //temp1 为row_1*col_1 需访问其col_row*(i+1)的元素 对应一维坐标为(col_result-1)*col_1+(i+1)
                //temp2 为row_2*col_2 需访问其(i+1)*col_result的元素 对应一维坐标为i*col_2+col_result
                //result[num] += temp1[(row_result - 1)*col_1 + i] * temp2[i*row_2 + col_result-(i+1)];
                result[num] += temp1[row_result * col_1 + i] * temp2[i * col_2 + col_result];
            }
            num++;
        }
    }
}

void evaluation::MultMatrix2(int row_1, int col_1, vector<double> &temp1, int row_2, int col_2, vector<double> &temp2,
                             vector<double> &result) {
//    int times = col_1;//times=col_1=row_2 为确定result某元素时进行的乘法(加法)次数
    int row_result, col_result;
    int num = 0;
    //num为一维数组形式的result的索引
    //row_result col_result为二维数组形式的result的索引
    for (row_result = 0; row_result < row_1; ++row_result) {
        for (col_result = 0; col_result < col_2; ++col_result) {
            result[num] = 0;
            for (int i = 0; i < row_2; ++i) {
                //对于m*n大小数组 如需访问其i*j元素 其对应的一维坐标为(i-1)*n+j
                //temp1 为row_1*col_1 需访问其col_row*(i+1)的元素 对应一维坐标为(col_result-1)*col_1+(i+1)
                //temp2 为row_2*col_2 需访问其(i+1)*col_result的元素 对应一维坐标为i*col_2+col_result
                //result[num] += temp1[(row_result - 1)*col_1 + i] * temp2[i*row_2 + col_result-(i+1)];
                result[num] += temp1[row_result * col_1 + i] * temp2[i * col_2 + col_result];
            }
            num++;
        }
    }
}

tensor &evaluation::get_result() {
    printf("evaluation.get_result key_expr_id:%d\n", key_expr_id);
//            printf("evaluation.get_result result:%d\n",variables_[key_expr_id]);
    return variables_[key_expr_id];
}


double &evaluation::get_result1() {
    printf("eval get_result first --------------- key= %s \n", keyStr.data());
    /*if (0 == key_op_type || 1 == key_op_type) {
        result_ = get_map(keyStr);
    }*/
    printf("eval->get_result Input after val %f\n", result_);
    return result_;
}

/*
double evaluation::get_map(string key) {
    printf("eval->Input k %s\n", key.c_str());
    double d = contextMap[key];
    printf("eval->Input d %f\n", d);
    return d;
}*/
