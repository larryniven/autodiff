#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "la/la.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    {"test-mul", []() {
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
    }},

    {"test-mul-grad", []() {
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

        auto r = get_child(t, 0);
        assert(r->grad != nullptr);

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
    }},

    {"test-logsoftmax", []() {
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
    }},

    {"test-conv", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        };

        la::vector<double> v {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> t {u.data(), {5, 5, 1}};
        la::weak_tensor<double> f {v.data(), {3, 3, 1, 1}};

        auto c = autodiff::corr(g.var(t), g.var(f));

        autodiff::eval(c, autodiff::eval_funcs);

        auto& o = autodiff::get_output<la::tensor_like<double>>(c);

        ebt::assert_equals(3, o.dim());
        ebt::assert_equals(5, o.size(0));
        ebt::assert_equals(5, o.size(1));
        ebt::assert_equals(1, o.size(2));

        ebt::assert_equals(9, o({0, 0, 0}));
        ebt::assert_equals(8, o({0, 1, 0}));
        ebt::assert_equals(7, o({0, 2, 0}));
    }},

    {"test-conv-grad", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        };

        la::vector<double> v {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        };

        la::vector<double> grad {
            1, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> u_t {u.data(), {5, 5, 1}};
        la::weak_tensor<double> v_t {v.data(), {3, 3, 1, 1}};
        la::weak_tensor<double> grad_t { grad.data(), {5, 5, 1}};

        auto var_u = g.var(u_t);
        auto var_v = g.var(v_t);
        auto c = autodiff::corr(var_u, var_v);
        autodiff::eval(c, autodiff::eval_funcs);

        auto& o = autodiff::get_output<la::tensor<double>>(c);
        double tmp = o({0, 0, 0});

        c->grad = std::make_shared<la::weak_tensor<double>>(grad_t);
        autodiff::grad(c, autodiff::grad_funcs);

        auto& grad_u = autodiff::get_grad<la::tensor_like<double>>(var_u);

        ebt::assert_equals(5, grad_u({0, 0, 0}));
        ebt::assert_equals(6, grad_u({0, 1, 0}));
        ebt::assert_equals(8, grad_u({1, 0, 0}));
        ebt::assert_equals(9, grad_u({1, 1, 0}));

        auto& grad_v = autodiff::get_grad<la::tensor_like<double>>(var_v);

        ebt::assert_equals(0, grad_v({0, 0, 0}));
        ebt::assert_equals(1, grad_v({2, 2, 0}));
    }}
};

int main()
{
    for (auto& t: tests) {
        std::cout << t.first << std::endl;
        t.second();
    }

    return 0;
}
