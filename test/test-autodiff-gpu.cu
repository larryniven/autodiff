#include "autodiff/autodiff-gpu.h"
#include "la/la-gpu.h"

std::vector<std::pair<std::string, std::function<void(void)>>> tests {
    {"test-dot", []() {
        la::vector<double> ha {1, 2, 3};
        la::gpu::vector<double> da { ha };

        la::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db { hb };

        autodiff::computation_graph g;
        auto output = autodiff::dot(g.var(da), g.var(db));

        autodiff::eval(output, autodiff::gpu::eval_funcs);
        ebt::assert_equals(14, autodiff::get_output<double>(output));
    }},

    {"test-mul", []() {
        la::matrix<double> ha {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        la::gpu::matrix<double> da { ha };

        la::vector<double> hb {1, 2, 3};
        la::gpu::vector<double> db { hb };

        autodiff::computation_graph g;
        auto output = autodiff::mul(g.var(da), g.var(db));

        autodiff::eval(output, autodiff::gpu::eval_funcs);

        la::vector<double> hc = to_host(autodiff::get_output<la::gpu::vector<double>>(output));
        ebt::assert_equals(3, hc.size());
        ebt::assert_equals(14, hc(0));
        ebt::assert_equals(32, hc(1));
        ebt::assert_equals(50, hc(2));
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
