#include "program.h"
#include "evaluation.h"
#include <string.h>
extern std::map<std::string, double> contextMap;
extern map<string, tensor> contextArrMap;

program::program() {
}

void program::append_expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int inputs[],
        int num_inputs) {
//    printf("prog.append_expression expr_id %d, op_name %s, op_type %s, inputs %d \n",
//            expr_id, op_name, op_type, num_inputs);
    expression *exp = new expression(expr_id, op_name, op_type, inputs, num_inputs);
    vec.push_back(*exp);
    key_expr_id = expr_id;
}

int program::add_op_param_double(
        const char *key,
        double value) {
//    printf("prog.add_op_param_double  key= %s, value= %f\n",
//           key, value);
//    const char *a = "value";
//    printf("key == value:%d, key_expr_id:%d\n", strncmp(key,a, strlen(key)),key_expr_id);
    /*if (strncmp(key,a, strlen(key)) == 0) {
//        printf("key == value\n");
        keyStr = key + key_expr_id;
    }else{
        keyStr = key;
    }*/
    keyStr = key + to_string(key_expr_id);
//    printf("prog.add_op_param_double --------------- key= %s \n", keyStr.data());
    contextMap[keyStr] = value;
    tensor *t = new tensor(value);
    contextArrMap[keyStr] = *t;
//    printf("prog.add_op_param_double   value= %f\n", contextMap[keyStr]);
//    printf("prog.add_op_param_double --------------- key= %s \n", keyStr.data());
    return 0;
}

int program::add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]) {
    /*printf("program.add_op_param_ndarray  key %s, value %p dim %d (",
           key, data, dim);
    for (int i = 0; i != dim; ++i)
        printf("%zu,", shape[i]);
    printf(")\n");*/
//    const char *a = "value";
//    printf("key == value:%d, key_expr_id:%d\n", strncmp(key,a, strlen(key)),key_expr_id);
    /*if (strncmp(key, a, strlen(key)) == 0) {
        printf("key == value\n");
        keyStr = to_string(key_expr_id);
    }else{
        keyStr = key;
    }*/
    keyStr = key + to_string(key_expr_id);
    printf("key:%s\n", keyStr.data());
    tensor *t = new tensor(dim, shape, data);
//    double b = t->data_[0];
//    printf("-----b:%f,contextArrMap.size:%d, keyStr:%s\n", b, contextArrMap.size(), &keyStr);
    contextArrMap[keyStr] = *t;
    return 0;
}

evaluation *program::build() {
    evaluation *eval = new evaluation(vec);
    eval->keyStr = keyStr;
    eval->key_expr_id = key_expr_id;
    return eval;
}
