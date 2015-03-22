#include "autodiff/autodiff.h"
#include "ebt/ebt.h"

std::vector<std::function<void(void)>> tests {
    []()
    {
        std::vector<double> x { 1, 2, 3 };

        std::vector<std::vector<double>> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        auto t = autodiff::mult(autodiff::var(A), autodiff::var(x));

        autodiff::eval(t, autodiff::eval_funcs);

        auto& result = autodiff::get_output<std::vector<double>>(t);

        ebt::assert_equals(14, result.at(0));
        ebt::assert_equals(32, result.at(1));
    },

    []()
    {
        std::vector<double> grad { 7, 8 };

        std::vector<double> x { 1, 2, 3 };

        std::vector<std::vector<double>> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        auto t = autodiff::mult(autodiff::var(A), autodiff::var(x));

        t->grad = std::make_shared<std::vector<double>>(grad);

        autodiff::grad(t, autodiff::grad_funcs);

        auto& A_grad = autodiff::get_grad<std::vector<std::vector<double>>>(t->children.at(0));
        auto& x_grad = autodiff::get_grad<std::vector<double>>(t->children.at(1));

        ebt::assert_equals(7, A_grad.at(0).at(0));
        ebt::assert_equals(14, A_grad.at(0).at(1));
        ebt::assert_equals(21, A_grad.at(0).at(2));

        ebt::assert_equals(8, A_grad.at(1).at(0));
        ebt::assert_equals(16, A_grad.at(1).at(1));
        ebt::assert_equals(24, A_grad.at(1).at(2));

        ebt::assert_equals(39, x_grad.at(0));
        ebt::assert_equals(54, x_grad.at(1));
        ebt::assert_equals(69, x_grad.at(2));
    }
};

int main()
{
    for (auto& t: tests) {
        t();
    }

    return 0;
}
