#include "autodiff/autodiff-gpu.h"
#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"

namespace autodiff {

    namespace gpu {

        void var_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {}

        void mul_eval(std::shared_ptr<op_t> t)
        {
            auto& A = get_output<la::gpu::matrix_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::vector<double> u;
                u.resize(A.rows());
                t->output = std::make_shared<la::gpu::vector<double>>(u);
            } else {
                auto& u = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(u);
            }

            auto& u = get_output<la::gpu::vector_like<double>>(t);
            la::gpu::mul(u, A, v);
        }

        void mul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto A_o = get_child(t, 0);
            auto v_o = get_child(t, 1);

            auto& A = get_output<la::gpu::matrix_like<double>>(A_o);
            auto& v = get_output<la::gpu::vector_like<double>>(v_o);

            if (A_o->grad == nullptr) {
                la::gpu::matrix<double> g;
                g.resize(grad.size(), v.size());
                A_o->grad = std::make_shared<la::gpu::matrix<double>>(std::move(g));
            }

            if (v_o->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(A.cols());
                v_o->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& A_grad = get_grad<la::gpu::matrix_like<double>>(A_o);
            auto& v_grad = get_grad<la::gpu::vector_like<double>>(v_o);

            autodiff::op::gpu::iouter_prod(A_grad, grad, v);
            autodiff::op::gpu::ilmul(v_grad, A, grad);
        }

        void mul_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& m = get_output<la::gpu::matrix_like<double>>(get_child(t, 0));
                double *d = mem.alloc(m.rows());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, m.rows()));
            }

            if (t->grad == nullptr) {
                auto& m = get_output<la::gpu::matrix_like<double>>(get_child(t, 0));
                double *d = mem.alloc(m.rows());
                la::gpu::weak_vector<double> g(d, m.rows());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void emul_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(u.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            la::gpu::emul(z, u, v);
        }

        void emul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto u_o = get_child(t, 0);
            auto v_o = get_child(t, 1);

            auto& u = get_output<la::gpu::vector_like<double>>(u_o);
            auto& v = get_output<la::gpu::vector_like<double>>(v_o);

            if (u_o->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(u.size());
                u_o->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            if (v_o->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(v.size());
                v_o->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::vector_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::vector_like<double>>(v_o);

            la::gpu::emul(u_grad, grad, v);
            la::gpu::emul(v_grad, grad, u);
        }

        void emul_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void logistic_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(v.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            autodiff::op::gpu::logistic(z, v);
        }

        void logistic_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);
            auto& output = get_output<la::gpu::vector_like<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(output.size());
                ch->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::vector_like<double>>(ch);
            autodiff::op::gpu::ilogistic_grad(result, grad, output);
        }

        void logistic_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void relu_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(v.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            autodiff::op::gpu::relu(z, v);
        }

        void relu_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector_like<double>>(t);
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(output.size());
                ch->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::vector_like<double>>(ch);
            autodiff::op::gpu::irelu_grad(result, grad, output);
        }

        void relu_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void tanh_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(v.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            autodiff::op::gpu::tanh(z, v);
        }

        void tanh_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);
            auto& output = get_output<la::gpu::vector_like<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(output.size());
                ch->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::vector_like<double>>(ch);
            autodiff::op::gpu::itanh_grad(result, grad, output);
        }

        void tanh_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void add_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            assert(g.adj[t->id].size() > 0);

#ifndef NDEBUG
            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                assert(get_output<la::gpu::vector_like<double>>(get_child(t, i-1)).size()
                    == get_output<la::gpu::vector_like<double>>(get_child(t, i)).size());
            }
#endif

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(get_output<la::gpu::vector<double>>(get_child(t, 0)).size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& result = get_output<la::gpu::vector_like<double>>(t);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto& u = get_output<la::gpu::vector_like<double>>(get_child(t, i));

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
                    la::gpu::vector<double> g;
                    g.resize(grad.size());
                    c->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
                }

                auto& u = get_grad<la::gpu::vector_like<double>>(c);

                la::gpu::iadd(u, grad);
            }
        }

        void add_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void sub_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::vector<double> g;
                g.resize(u.size());
                t->output = std::make_shared<la::gpu::vector<double>>(g);
            } else {
                auto& w = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(w);
            }

            auto& result = get_output<la::gpu::vector_like<double>>(t);

            la::gpu::copy(result, u);
            la::gpu::isub(result, v);
        }

        void sub_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto u_o = get_child(t, 0);
            if (u_o->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(grad.size());
                u_o->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto v_o = get_child(t, 1);
            if (v_o->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(grad.size());
                v_o->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::vector_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::vector_like<double>>(v_o);

            la::gpu::iadd(u_grad, grad);
            la::gpu::isub(v_grad, grad);
        }

        void sub_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void softmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(v.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            autodiff::op::gpu::softmax(z, v);
        }

        void softmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector_like<double>>(t);
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(output.size());
                ch->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::vector_like<double>>(ch);
            autodiff::op::gpu::isoftmax_grad(result, grad, output);
        }

        void softmax_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void logsoftmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::vector<double> z;
                z.resize(v.size());
                t->output = std::make_shared<la::gpu::vector<double>>(z);
            } else {
                auto& z = get_output<la::gpu::vector_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::vector_like<double>>(t);
            autodiff::op::gpu::logsoftmax(z, v);
        }

        void logsoftmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::vector_like<double>>(t);
            auto& grad = get_grad<la::gpu::vector_like<double>>(t);

            auto ch = get_child(t, 0);
            if (ch->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(output.size());
                ch->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::vector_like<double>>(ch);
            autodiff::op::gpu::ilogsoftmax_grad(result, grad, output);
        }

        void logsoftmax_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem)
        {
            if (t->output == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                t->output = std::make_shared<la::gpu::weak_vector<double>>(
                    la::gpu::weak_vector<double>(d, v.size()));
            }

            if (t->grad == nullptr) {
                auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
                double *d = mem.alloc(v.size());
                la::gpu::weak_vector<double> g(d, v.size());
                la::gpu::zero(g);
                t->grad = std::make_shared<la::gpu::weak_vector<double>>(g);
            }
        }

        void dot_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::vector_like<double>>(get_child(t, 0));
            auto& u = get_output<la::gpu::vector_like<double>>(get_child(t, 1));

            t->output = std::make_shared<double>(la::gpu::dot(v, u));
        }

        void dot_grad(std::shared_ptr<op_t> t)
        {
            auto c0 = get_child(t, 0);
            auto c1 = get_child(t, 1);

            auto& v = get_output<la::gpu::vector_like<double>>(c0);
            auto& u = get_output<la::gpu::vector_like<double>>(c1);

            assert(v.size() == u.size());

            double grad = get_grad<double>(t);

            if (c0->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(u.size());
                c0->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& v_grad = get_grad<la::gpu::vector_like<double>>(c0);

            cublasDaxpy(la::gpu::device::get_handle(), v_grad.size(), &grad, u.data(), 1, v_grad.data(), 1);

            if (c1->grad == nullptr) {
                la::gpu::vector<double> g;
                g.resize(v.size());
                c1->grad = std::make_shared<la::gpu::vector<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::vector_like<double>>(c1);

            cublasDaxpy(la::gpu::device::get_handle(), u_grad.size(), &grad, v.data(), 1, u_grad.data(), 1);
        }

        void dot_alloc(std::shared_ptr<autodiff::op_t> t, memory_pool<double>& mem)
        {}

        void alloc_vertex(std::shared_ptr<autodiff::op_t> const& t,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs)
        {
            funcs.at(t->name)(t, mem);
        }

        void alloc(std::vector<std::shared_ptr<op_t>> const& topo_order,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs)
        {
            for (int i = topo_order.size() - 1; i >= 0; --i) {
                alloc_vertex(topo_order[i], mem, funcs);
            }
        }

        void alloc(std::shared_ptr<op_t> const& root,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs)
        {
            std::vector<std::shared_ptr<autodiff::op_t>> order = topo_order(root);
            alloc(order, mem, funcs);
        }

    }
}
