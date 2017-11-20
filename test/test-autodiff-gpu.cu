#include "autodiff/autodiff-gpu.h"
#include "la/la-gpu.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    // {"test-dot", []() {
    //     la::vector<double> ha {1, 2, 3};
    //     la::gpu::vector<double> da { ha };

    //     la::vector<double> hb {1, 2, 3};
    //     la::gpu::vector<double> db { hb };

    //     autodiff::interpreter& itp = autodiff::interpreter::get_instance();
    //     itp.eval_funcs = autodiff::gpu::eval_funcs;
    //     itp.grad_funcs = autodiff::gpu::grad_funcs;

    //     autodiff::computation_graph g;
    //     auto output = autodiff::dot(g.var(da), g.var(db));

    //     ebt::assert_equals(14, autodiff::get_output<double>(output));
    // }},

    {"test-mul", []() {
        la::cpu::vector<double> ha {1, 2, 3, 4, 5, 6, 7, 8, 9};
        la::gpu::vector<double> da {ha};
        la::gpu::tensor<double> dat { da, {3, 3} };

        la::cpu::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db {hb};
        la::gpu::tensor<double> dbt { db, {3}};

        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        autodiff::computation_graph g;
        auto output = autodiff::mul(g.var(dbt), g.var(dat));

        la::cpu::tensor<double> hc = to_host(autodiff::get_output<la::gpu::tensor_like<double>>(output));

        ebt::assert_equals(3, hc.vec_size());
        ebt::assert_equals(30, hc({0}));
        ebt::assert_equals(36, hc({1}));
        ebt::assert_equals(42, hc({2}));
    }},

    {"test-rep-row-to", []() {
        la::cpu::vector<double> ha {1, 2, 3};
        la::gpu::vector<double> da {ha};
        la::gpu::tensor<double> dat {da, {3}};

        la::cpu::vector<double> hb {4, 5, 6, 7, 8, 9};
        la::gpu::vector<double> db {hb};
        la::gpu::tensor<double> dbt { db, {2, 3}};

        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        autodiff::computation_graph g;
        auto output = autodiff::rep_row_to(g.var(dat), g.var(dbt));

        la::cpu::tensor<double> hc = to_host(autodiff::get_output<la::gpu::tensor_like<double>>(output));
        ebt::assert_equals(6, hc.vec_size());
        ebt::assert_equals(1, hc({0, 0}));
        ebt::assert_equals(2, hc({0, 1}));
        ebt::assert_equals(3, hc({0, 2}));
        ebt::assert_equals(1, hc({1, 0}));
        ebt::assert_equals(2, hc({1, 1}));
        ebt::assert_equals(3, hc({1, 2}));
    }},

    {"test-rep-row-to-grad", []() {
        la::cpu::vector<double> ha {1, 2, 3};
        la::gpu::vector<double> da {ha};
        la::gpu::tensor<double> dat {da, {3}};

        la::cpu::vector<double> hb {4, 5, 6, 7, 8, 9};
        la::gpu::vector<double> db {hb};
        la::gpu::tensor<double> dbt { db, {2, 3}};

        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        autodiff::computation_graph g;
        auto t = g.var(dat);
        auto output = autodiff::rep_row_to(t, g.var(dbt));

        la::cpu::vector<double> hg {10, 11, 12, 13, 14, 15};
        la::gpu::vector<double> dg {hg};
        la::gpu::tensor<double> dgt {dg, {2, 3}};

        output->grad = std::make_shared<la::gpu::tensor<double>>(dgt);
        autodiff::eval_vertex(output, autodiff::gpu::grad_funcs);

        la::cpu::tensor<double> hc = to_host(autodiff::get_grad<la::gpu::tensor_like<double>>(t));
        ebt::assert_equals(3, hc.vec_size());
        ebt::assert_equals(23, hc({0}));
        ebt::assert_equals(25, hc({1}));
        ebt::assert_equals(27, hc({2}));
    }},

    {"test-rep-col-to", []() {
        la::cpu::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};
        la::gpu::tensor<double> dat {da, {2}};

        la::cpu::vector<double> hb {4, 5, 6, 7, 8, 9};
        la::gpu::vector<double> db {hb};
        la::gpu::tensor<double> dbt { db, {2, 3}};

        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        autodiff::computation_graph g;
        auto output = autodiff::rep_col_to(g.var(dat), g.var(dbt));

        la::cpu::tensor<double> hc = to_host(autodiff::get_output<la::gpu::tensor_like<double>>(output));
        ebt::assert_equals(6, hc.vec_size());
        ebt::assert_equals(1, hc({0, 0}));
        ebt::assert_equals(1, hc({0, 1}));
        ebt::assert_equals(1, hc({0, 2}));
        ebt::assert_equals(2, hc({1, 0}));
        ebt::assert_equals(2, hc({1, 1}));
        ebt::assert_equals(2, hc({1, 2}));
    }},

    {"test-rep-col-to-grad", []() {
        la::cpu::vector<double> ha {1, 2};
        la::gpu::vector<double> da {ha};
        la::gpu::tensor<double> dat {da, {2}};

        la::cpu::vector<double> hb {4, 5, 6, 7, 8, 9};
        la::gpu::vector<double> db {hb};
        la::gpu::tensor<double> dbt { db, {2, 3}};

        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        autodiff::computation_graph g;
        auto t = g.var(dat);
        auto output = autodiff::rep_col_to(t, g.var(dbt));

        la::cpu::vector<double> hg {10, 11, 12, 13, 14, 15};
        la::gpu::vector<double> dg {hg};
        la::gpu::tensor<double> dgt {dg, {2, 3}};

        output->grad = std::make_shared<la::gpu::tensor<double>>(dgt);
        autodiff::eval_vertex(output, autodiff::gpu::grad_funcs);

        la::cpu::tensor<double> hc = to_host(autodiff::get_grad<la::gpu::tensor_like<double>>(t));
        ebt::assert_equals(2, hc.vec_size());
        ebt::assert_equals(33, hc({0}));
        ebt::assert_equals(42, hc({1}));
    }},

    {"test-logsoftmax", []() {
        autodiff::interpreter& itp = autodiff::interpreter::get_instance();
        itp.eval_funcs = autodiff::gpu::eval_funcs;
        itp.grad_funcs = autodiff::gpu::grad_funcs;

        la::cpu::vector<double> x { 1, 2, 3 };
        la::cpu::vector<double> expected {
            -2.4076059644438, -1.4076059644438, -0.4076059644438 };

        la::gpu::tensor<double> hx {la::gpu::vector<double>(x)};

        autodiff::computation_graph g;

        auto t = autodiff::logsoftmax(g.var(hx));
        la::cpu::tensor<double> result = la::gpu::to_host(autodiff::get_output<la::gpu::tensor_like<double>>(t));

        ebt::assert_equals(expected(0), result({0}));
        ebt::assert_equals(expected(1), result({1}));
        ebt::assert_equals(expected(2), result({2}));

#if 0
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
#endif
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
