#include <stdio.h>
#include "src/libeasynn.h"

int main()
{
    program *prog = create_program();

    int inputs0[] = {};
    append_expression(prog, 0, "a", "Input", inputs0, 0);

    int inputs1[] = {};
    append_expression(prog, 1, "b", "Input", inputs1, 2);

    int inputs2[] = {0,1};
    append_expression(prog, 2, "", "Add", inputs2, 2);

    evaluation *eval = build(prog);
    add_kwargs_double(eval, "a", 5);
    size_t myShape[2] = {2, 3};
    size_t myShape2[2] = {3,2};
    double myData[3] = {1,2,3};
    add_kwargs_ndarray(eval, "a", 2, myShape, myData);
    add_kwargs_ndarray(eval, "b", 2, myShape2, myData);


    int dim = 0;
    size_t *shape = nullptr;
    double *data = nullptr;
    if (execute(eval, &dim, &shape, &data) != 0)
    {
        printf("evaluation fails\n");
        return -1;
    }

    if (dim == 0)
        printf("res = %f\n", data[0]);
    else
        printf("result as tensor is not supported yet\n");

    return 0;
}