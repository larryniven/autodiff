#include "autodiff/autodiff-gpu.h"
#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"

namespace autodiff {

    namespace gpu {

        void mult_eval(std::shared_ptr<op_t> t)
        {
            auto& A = get_output<la::gpu::matrix<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 1));

            t->output = std::make_shared<la::gpu::vector<double>>(la::gpu::mult(A, v));
        }

        void mult_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector<double>>(t);

            auto A_o = get_child(t, 0);
            auto v_o = get_child(t, 1);

            auto& A = get_output<la::gpu::matrix<double>>(A_o);
            auto& v = get_output<la::gpu::vector<double>>(v_o);

            if (A_o->grad == nullptr) {
                A_o->grad = std::make_shared<la::gpu::matrix<double>>(la::gpu::matrix<double>());
            }

            if (v_o->grad == nullptr) {
                v_o->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& A_grad = get_grad<la::gpu::matrix<double>>(A_o);
            auto& v_grad = get_grad<la::gpu::vector<double>>(v_o);

            autodiff::op::gpu::iouter_prod(A_grad, grad, v);
            autodiff::op::gpu::ilmult(v_grad, A, grad);
        }

        void logistic_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));

            t->output = std::make_shared<la::gpu::vector<double>>(autodiff::op::gpu::logistic(v));
        }

        void logistic_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector<double>>(t);
            auto& output = get_output<la::gpu::vector<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                ch->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_grad<la::gpu::vector<double>>(ch);
            autodiff::op::gpu::ilogistic_grad(result, grad, output);
        }

        void relu_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));

            t->output = std::make_shared<la::gpu::vector<double>>(autodiff::op::gpu::relu(v));
        }

        void relu_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector<double>>(t);
            auto& grad = get_grad<la::gpu::vector<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                ch->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_grad<la::gpu::vector<double>>(ch);
            autodiff::op::gpu::irelu_grad(result, grad, output);
        }

        void tanh_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));

            t->output = std::make_shared<la::gpu::vector<double>>(autodiff::op::gpu::tanh(v));
        }

        void tanh_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector<double>>(t);
            auto& output = get_output<la::gpu::vector<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                ch->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_grad<la::gpu::vector<double>>(ch);
            autodiff::op::gpu::itanh_grad(result, grad, output);
        }

        void add_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            assert(g.adj[t->id].size() > 0);

#ifndef NDEBUG
            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                assert(get_output<la::gpu::vector<double>>(get_child(t, i-1)).size()
                    == get_output<la::gpu::vector<double>>(get_child(t, i)).size());
            }
#endif

            if (t->output == nullptr) {
                t->output = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_output<la::gpu::vector<double>>(t);
            result.resize(get_output<la::gpu::vector<double>>(get_child(t, 0)).size());

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto& u = get_output<la::gpu::vector<double>>(get_child(t, i));

                la::gpu::iadd(result, u);
            }
        }

        void add_grad(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            auto& grad = get_grad<la::gpu::vector<double>>(t);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto c = get_child(t, i);

                if (c->grad == nullptr) {
                    c->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
                }

                auto& u = get_grad<la::gpu::vector<double>>(c);
                u.resize(grad.size());

                la::gpu::iadd(u, grad);
            }
        }

        void softmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));

            t->output = std::make_shared<la::gpu::vector<double>>(autodiff::op::gpu::softmax(v));
        }

        void softmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector<double>>(t);
            auto& grad = get_grad<la::gpu::vector<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                ch->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_grad<la::gpu::vector<double>>(ch);
            autodiff::op::gpu::isoftmax_grad(result, grad, output);
        }

        void logsoftmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));

            t->output = std::make_shared<la::gpu::vector<double>>(autodiff::op::gpu::logsoftmax(v));
        }

        void logsoftmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector<double>>(t);
            auto& grad = get_grad<la::gpu::vector<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                ch->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& result = get_grad<la::gpu::vector<double>>(ch);
            autodiff::op::gpu::ilogsoftmax_grad(result, grad, output);
        }

        void dot_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector<double>>(get_child(t, 0));
            auto& u = get_output<la::gpu::vector<double>>(get_child(t, 1));

            t->output = std::make_shared<double>(la::gpu::dot(v, u));
        }

        void dot_grad(std::shared_ptr<op_t> t)
        {
            auto c0 = get_child(t, 0);
            auto c1 = get_child(t, 1);

            auto& v = get_output<la::gpu::vector<double>>(c0);
            auto& u = get_output<la::gpu::vector<double>>(c1);

            assert(v.size() == u.size());

            double grad = get_grad<double>(t);

            if (c0->grad == nullptr) {
                c0->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& v_grad = get_grad<la::gpu::vector<double>>(c0);
            v_grad.resize(u.size());

            cublasDaxpy(la::gpu::device::get_handle(), v_grad.size(), &grad, u.data(), 1, v_grad.data(), 1);

            if (c1->grad == nullptr) {
                c1->grad = std::make_shared<la::gpu::vector<double>>(la::gpu::vector<double>());
            }

            auto& u_grad = get_grad<la::gpu::vector<double>>(c1);
            u_grad.resize(v.size());

            cublasDaxpy(la::gpu::device::get_handle(), u_grad.size(), &grad, v.data(), 1, u_grad.data(), 1);
        }

    }
}