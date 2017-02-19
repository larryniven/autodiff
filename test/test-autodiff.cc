#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "la/la.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    {"test-mul", []() {
        la::vector<double> x { 1, 2 };

        la::matrix<double> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        autodiff::computation_graph g;

        auto x_var = g.var(la::weak_tensor<double> { x });
        auto A_var = g.var(la::weak_tensor<double> { A });

        auto t = autodiff::mul(x_var, A_var);

        // autodiff::eval(t, autodiff::eval_funcs);

        la::tensor_like<double>& result = autodiff::get_output<la::tensor_like<double>>(t);

        ebt::assert_equals(9, result({0}));
        ebt::assert_equals(12, result({1}));
        ebt::assert_equals(15, result({2}));
    }},

    {"test-mul-grad", []() {
        la::vector<double> grad { 7, 8, 9 };

        la::vector<double> x { 1, 2 };

        la::matrix<double> A {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        autodiff::computation_graph g;

        auto t = autodiff::mul(g.var(la::weak_tensor<double>{ x }), g.var(la::weak_tensor<double>{ A }));

        t->grad = std::make_shared<la::weak_tensor<double>>(la::weak_tensor<double> { grad });

        autodiff::grad(t, autodiff::grad_funcs);

        auto r = get_child(t, 0);
        assert(r->grad != nullptr);

        la::weak_vector<double> x_grad = autodiff::get_grad<la::tensor_like<double>>(get_child(t, 0)).as_vector();
        la::weak_matrix<double> A_grad = autodiff::get_grad<la::tensor_like<double>>(get_child(t, 1)).as_matrix();

        ebt::assert_equals(50, x_grad(0));
        ebt::assert_equals(122, x_grad(1));

        ebt::assert_equals(7, A_grad(0, 0));
        ebt::assert_equals(8, A_grad(0, 1));
        ebt::assert_equals(9, A_grad(0, 2));

        ebt::assert_equals(14, A_grad(1, 0));
        ebt::assert_equals(16, A_grad(1, 1));
        ebt::assert_equals(18, A_grad(1, 2));
    }},

    {"test-logsoftmax", []() {
        la::vector<double> x { 1, 2, 3 };
        la::vector<double> expected {
            -2.4076059644438, -1.4076059644438, -0.4076059644438 };

        autodiff::computation_graph g;

        auto t = autodiff::logsoftmax(g.var(la::weak_tensor<double>{ x }));
        // autodiff::eval(t, autodiff::eval_funcs);
        la::weak_vector<double> result = autodiff::get_output<la::tensor_like<double>>(t).as_vector();

        ebt::assert_equals(expected(0), result(0));
        ebt::assert_equals(expected(1), result(1));
        ebt::assert_equals(expected(2), result(2));

        {
            la::vector<double> grad { 1, 0, 0 };
            la::vector<double> grad_expected {
                0.909969426716728, -0.24472847121259633, -0.665240955655122 };

            t->grad = std::make_shared<la::weak_tensor<double>>(la::weak_tensor<double>{ grad });
            autodiff::grad(t, autodiff::grad_funcs);

            la::weak_vector<double> grad_result = autodiff::get_grad<la::tensor_like<double>>(
                autodiff::get_child(t, 0)).as_vector();

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }

        {
            autodiff::clear_grad(t);

            la::vector<double> grad { 0, 1, 0 };
            la::vector<double> grad_expected {
                -0.09003057328316189, 0.7552715287884038, -0.665240955655122 };

            t->grad = std::make_shared<la::weak_tensor<double>>(la::weak_tensor<double>{ grad });
            autodiff::grad(t, autodiff::grad_funcs);

            la::weak_vector<double> grad_result = autodiff::get_grad<la::tensor_like<double>>(
                autodiff::get_child(t, 0)).as_vector();

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }

        {
            autodiff::clear_grad(t);

            la::vector<double> grad { 0, 0, 1 };
            la::vector<double> grad_expected {
                -0.09003057328316189, -0.24472847121259633, 0.33475904434698833 };

            t->grad = std::make_shared<la::weak_tensor<double>>(la::weak_tensor<double>{ grad });
            autodiff::grad(t, autodiff::grad_funcs);

            la::weak_vector<double> grad_result = autodiff::get_grad<la::tensor_like<double>>(
                autodiff::get_child(t, 0)).as_vector();

            ebt::assert_equals(grad_expected(0), grad_result(0));
            ebt::assert_equals(grad_expected(1), grad_result(1));
            ebt::assert_equals(grad_expected(2), grad_result(2));
        }
    }},

    {"test-2d-conv", []() {
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

        la::weak_tensor<double> u_t {u.data(), {5, 5, 1}};
        la::weak_tensor<double> v_t {v.data(), {3, 3, 1, 1}};

        auto c = autodiff::corr(g.var(u_t), g.var(v_t));

        // autodiff::eval(c, autodiff::eval_funcs);

        auto& o = autodiff::get_output<la::tensor_like<double>>(c);

        ebt::assert_equals(3, o.dim());
        ebt::assert_equals(5, o.size(0));
        ebt::assert_equals(5, o.size(1));
        ebt::assert_equals(1, o.size(2));

        ebt::assert_equals(9, o({0, 0, 0}));
        ebt::assert_equals(8, o({0, 1, 0}));
        ebt::assert_equals(7, o({0, 2, 0}));
    }},

    {"test-2d-conv-grad", []() {
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
        // autodiff::eval(c, autodiff::eval_funcs);

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

        ebt::assert_equals(0, grad_v({0, 0, 0, 0}));
        ebt::assert_equals(1, grad_v({2, 2, 0, 0}));
    }},

    {"test-3d-conv", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 1, 1, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        la::vector<double> v {
            1, 10, 2, 11, 3, 12,

            4, 13, 5, 14, 6, 15,

            7, 16, 8, 17, 9, 18
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> t {u.data(), {5, 5, 2}};
        la::weak_tensor<double> f {v.data(), {3, 3, 2, 1}};

        auto c = autodiff::corr(g.var(t), g.var(f));

        // autodiff::eval(c, autodiff::eval_funcs);

        auto& o = autodiff::get_output<la::tensor_like<double>>(c);

        ebt::assert_equals(3, o.dim());
        ebt::assert_equals(5, o.size(0));
        ebt::assert_equals(5, o.size(1));
        ebt::assert_equals(1, o.size(2));

        ebt::assert_equals(27, o({0, 0, 0}));
        ebt::assert_equals(25, o({0, 1, 0}));
        ebt::assert_equals(23, o({0, 2, 0}));
    }},

    {"test-3d-conv-grad", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 1, 1, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        la::vector<double> v {
            1, 10, 2, 11, 3, 12,

            4, 13, 5, 14, 6, 15,

            7, 16, 8, 17, 9, 18
        };

        la::vector<double> grad {
            1, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> u_t {u.data(), {5, 5, 2}};
        la::weak_tensor<double> v_t {v.data(), {3, 3, 2, 1}};

        auto var_u = g.var(u_t);
        auto var_v = g.var(v_t);
        auto c = autodiff::corr(var_u, var_v);

        // autodiff::eval(c, autodiff::eval_funcs);

        la::weak_tensor<double> grad_t { grad.data(), {5, 5, 1} };
        c->grad = std::make_shared<la::weak_tensor<double>>(grad_t);
        autodiff::grad(c, autodiff::grad_funcs);

        auto& grad_u = autodiff::get_grad<la::tensor_like<double>>(var_u);

        ebt::assert_equals(5, grad_u({0, 0, 0}));
        ebt::assert_equals(14, grad_u({0, 0, 1}));
        ebt::assert_equals(9, grad_u({1, 1, 0}));
        ebt::assert_equals(18, grad_u({1, 1, 1}));

        auto& grad_v = autodiff::get_grad<la::tensor_like<double>>(var_v);

        ebt::assert_equals(0, grad_v({0, 0, 0, 0}));
        ebt::assert_equals(0, grad_v({0, 0, 1, 0}));
        ebt::assert_equals(1, grad_v({2, 2, 0, 0}));
        ebt::assert_equals(1, grad_v({2, 2, 1, 0}));
    }},

    {"test-3d-multi-conv", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 1, 1, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        la::vector<double> v {
            1, 19, 10, 28,
            2, 20, 11, 29,
            3, 21, 12, 30,
            4, 22, 13, 31,
            5, 23, 14, 32,
            6, 24, 15, 33,
            7, 25, 16, 34,
            8, 26, 17, 35,
            9, 27, 18, 36,
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> t {u.data(), {5, 5, 2}};
        la::weak_tensor<double> f {v.data(), {3, 3, 2, 2}};

        auto c = autodiff::corr(g.var(t), g.var(f));

        // autodiff::eval(c, autodiff::eval_funcs);

        auto& o = autodiff::get_output<la::tensor_like<double>>(c);

        ebt::assert_equals(3, o.dim());
        ebt::assert_equals(5, o.size(0));
        ebt::assert_equals(5, o.size(1));
        ebt::assert_equals(2, o.size(2));

        ebt::assert_equals(27, o({0, 0, 0}));
        ebt::assert_equals(25, o({0, 1, 0}));
        ebt::assert_equals(23, o({0, 2, 0}));

        ebt::assert_equals(63, o({0, 0, 1}));
        ebt::assert_equals(61, o({0, 1, 1}));
        ebt::assert_equals(59, o({0, 2, 1}));
    }},

    {"test-3d-multi-conv-grad", []() {
        la::vector<double> u {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 1, 1, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        la::vector<double> v {
            1, 19, 10, 28,
            2, 20, 11, 29,
            3, 21, 12, 30,
            4, 22, 13, 31,
            5, 23, 14, 32,
            6, 24, 15, 33,
            7, 25, 16, 34,
            8, 26, 17, 35,
            9, 27, 18, 36,
        };

        la::vector<double> grad {
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 

            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        autodiff::computation_graph g;

        la::weak_tensor<double> u_t {u.data(), {5, 5, 2}};
        la::weak_tensor<double> v_t {v.data(), {3, 3, 2, 2}};

        auto var_u = g.var(u_t);
        auto var_v = g.var(v_t);
        auto c = autodiff::corr(var_u, var_v);

        // autodiff::eval(c, autodiff::eval_funcs);

        la::weak_tensor<double> grad_t { grad.data(), {5, 5, 2} };
        c->grad = std::make_shared<la::weak_tensor<double>>(grad_t);

        autodiff::grad(c, autodiff::grad_funcs);

        auto& grad_u = autodiff::get_grad<la::tensor_like<double>>(var_u);

        ebt::assert_equals(23, grad_u({0, 0, 0}));
        ebt::assert_equals(24, grad_u({0, 1, 0}));
        ebt::assert_equals(27, grad_u({1, 1, 0}));

        ebt::assert_equals(32, grad_u({0, 0, 1}));
        ebt::assert_equals(33, grad_u({0, 1, 1}));
        ebt::assert_equals(36, grad_u({1, 1, 1}));

    }},

    {"test-dropout-mask", []() {
        la::vector<double> u {
            0, 0, 0,
            0, 0, 0
        };

        la::weak_tensor<double> u_t { u.data(), {2, 3} };

        std::default_random_engine gen;

        autodiff::computation_graph g;

        auto op = autodiff::dropout_mask(g.var(u_t), 0.0, gen);

        // autodiff::eval(op, autodiff::eval_funcs);

        auto& result = autodiff::get_output<la::tensor_like<double>>(op);

        ebt::assert_equals(6, result.vec_size());
        ebt::assert_equals(2, result.dim());
        ebt::assert_equals(2, result.size(0));
        ebt::assert_equals(3, result.size(1));
        ebt::assert_equals(1, result({0, 0}));
        ebt::assert_equals(1, result({0, 1}));
        ebt::assert_equals(1, result({0, 2}));

    }},

    {"test-emul", []() {
        la::vector<double> u {
            1, 2, 3,
            4, 5, 6
        };

        la::vector<double> v {
            7, 8, 9,
            10, 11, 12
        };

        la::weak_tensor<double> u_t { u.data(), {2, 3} };
        la::weak_tensor<double> v_t { v.data(), {2, 3} };

        autodiff::computation_graph g;

        auto op = autodiff::emul(g.var(u_t), g.var(v_t));

        // autodiff::eval(op, autodiff::eval_funcs);

        auto& result = autodiff::get_output<la::tensor_like<double>>(op);

        ebt::assert_equals(2, result.dim());
        ebt::assert_equals(2, result.size(0));
        ebt::assert_equals(3, result.size(1));
        ebt::assert_equals(7, result({0, 0}));
        ebt::assert_equals(16, result({0, 1}));
        ebt::assert_equals(27, result({0, 2}));
        ebt::assert_equals(40, result({1, 0}));

    }},

    {"test-emul-dropout", []() {
        la::vector<double> u {
            1, 2, 3,
            4, 5, 6
        };

        la::weak_tensor<double> u_t { u.data(), {2, 3} };

        std::default_random_engine gen;

        autodiff::computation_graph g;

        auto u_op = g.var(u_t);
        auto op = autodiff::emul(u_op, autodiff::dropout_mask(u_op, 0.0, gen));

        // autodiff::eval(op, autodiff::eval_funcs);

        auto& result = autodiff::get_output<la::tensor_like<double>>(op);

        ebt::assert_equals(2, result.dim());
        ebt::assert_equals(2, result.size(0));
        ebt::assert_equals(3, result.size(1));
        ebt::assert_equals(1, result({0, 0}));
        ebt::assert_equals(2, result({0, 1}));
        ebt::assert_equals(3, result({0, 2}));
        ebt::assert_equals(4, result({1, 0}));

    }},
};

int main()
{
    for (auto& t: tests) {
        std::cout << t.first << std::endl;
        t.second();
    }

    return 0;
}
