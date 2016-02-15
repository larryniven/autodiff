#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "la/la.h"

std::vector<std::function<void(void)>> tests {
    []()
    {
        la::vector<double> x { 1, 2, 3 };

        la::matrix<double> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        autodiff::computation_graph g;

        auto t = autodiff::mul(g.var(A), g.var(x));

        autodiff::eval(t, autodiff::eval_funcs);

        auto& result = autodiff::get_output<la::vector_like<double>>(t);

        ebt::assert_equals(14, result(0));
        ebt::assert_equals(32, result(1));
    },

    []()
    {
        la::vector<double> grad { 7, 8 };

        la::vector<double> x { 1, 2, 3 };

        la::matrix<double> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        autodiff::computation_graph g;

        auto t = autodiff::mul(g.var(A), g.var(x));

        t->grad = std::make_shared<la::vector<double>>(grad);

        autodiff::grad(t, autodiff::grad_funcs);

        auto& A_grad = autodiff::get_grad<la::matrix_like<double>>(get_child(t, 0));
        auto& x_grad = autodiff::get_grad<la::vector_like<double>>(get_child(t, 1));

        ebt::assert_equals(7, A_grad(0, 0));
        ebt::assert_equals(14, A_grad(0, 1));
        ebt::assert_equals(21, A_grad(0, 2));

        ebt::assert_equals(8, A_grad(1, 0));
        ebt::assert_equals(16, A_grad(1, 1));
        ebt::assert_equals(24, A_grad(1, 2));

        ebt::assert_equals(39, x_grad(0));
        ebt::assert_equals(54, x_grad(1));
        ebt::assert_equals(69, x_grad(2));
    },

    []()
    {
        la::vector<double> x { 1, 2, 3 };
        la::vector<double> expected {
            -2.4076059644438, -1.4076059644438, -0.4076059644438 };

        autodiff::computation_graph g;

        auto t = autodiff::logsoftmax(g.var(x));
        autodiff::eval(t, autodiff::eval_funcs);
        auto& result = autodiff::get_output<la::vector_like<double>>(t);

        ebt::assert_equals(expected(0), result(0));
        ebt::assert_equals(expected(1), result(1));
        ebt::assert_equals(expected(2), result(2));

        {
            la::vector<double> grad { 1, 0, 0 };
            la::vector<double> grad_expected {
                0.909969426716728, -0.24472847121259633, -0.665240955655122 };

            t->grad = std::make_shared<la::vector<double>>(grad);
            autodiff::grad(t, autodiff::grad_funcs);

            auto& grad_result = autodiff::get_grad<la::vector_like<double>>(autodiff::get_child(t, 0));

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }

        {
            autodiff::clear_grad(t);

            la::vector<double> grad { 0, 1, 0 };
            la::vector<double> grad_expected {
                -0.09003057328316189, 0.7552715287884038, -0.665240955655122 };

            t->grad = std::make_shared<la::vector<double>>(grad);
            autodiff::grad(t, autodiff::grad_funcs);

            auto& grad_result = autodiff::get_grad<la::vector_like<double>>(autodiff::get_child(t, 0));

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }

        {
            autodiff::clear_grad(t);

            la::vector<double> grad { 0, 0, 1 };
            la::vector<double> grad_expected {
                -0.09003057328316189, -0.24472847121259633, 0.33475904434698833 };

            t->grad = std::make_shared<la::vector<double>>(grad);
            autodiff::grad(t, autodiff::grad_funcs);

            auto& grad_result = autodiff::get_grad<la::vector_like<double>>(autodiff::get_child(t, 0));

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }
    }
};

int main()
{
    for (auto& t: tests) {
        t();
    }

    return 0;
}
