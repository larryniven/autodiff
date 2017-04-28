#include "autodiff/autodiff-gpu.h"
#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

namespace autodiff {

    namespace gpu {

        void mul_eval(std::shared_ptr<op_t> t)
        {
            auto& a = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& b = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> c;

                std::vector<unsigned int> sizes = a.sizes();
                sizes.pop_back();
                sizes.push_back(b.size(b.dim() - 1));

                c.resize(sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(c);
            } else {
                auto& c = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(c);
            }

            auto& c = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::mul(c, a, b);
        }

        void mul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto a_o = get_child(t, 0);
            auto b_o = get_child(t, 1);

            auto& a = get_output<la::gpu::tensor_like<double>>(a_o);
            auto& b = get_output<la::gpu::tensor_like<double>>(b_o);

            if (a_o->grad_needed && a_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, a);
                a_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (b_o->grad_needed && b_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, b);
                b_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& a_grad = get_grad<la::gpu::tensor_like<double>>(a_o);
            auto& b_grad = get_grad<la::gpu::tensor_like<double>>(b_o);

            if (a_o->grad_needed) {
                la::gpu::rtmul(a_grad, grad, b);
            }

            if (b_o->grad_needed) {
                la::gpu::ltmul(b_grad, a, grad);
            }
        }

        void emul_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, u);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);
            la::gpu::emul(z, u, v);
        }

        void emul_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto u_o = get_child(t, 0);
            auto v_o = get_child(t, 1);

            auto& u = get_output<la::gpu::tensor_like<double>>(u_o);
            auto& v = get_output<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed && u_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            if (v_o->grad_needed && v_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v);
                v_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed) {
                la::gpu::emul(u_grad, grad, v);
            }

            if (v_o->grad_needed) {
                la::gpu::emul(v_grad, grad, u);
            }
        }

        void logistic_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);
            op::gpu::logistic(z, v);
        }

        void logistic_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);
            auto& output = get_output<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                op::gpu::ilogistic_grad(result, grad, output);
            }
        }

        void tanh_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);
            op::gpu::tanh(z, v);
        }

        void tanh_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);
            auto& output = get_output<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                op::gpu::itanh_grad(result, grad, output);
            }
        }

        void add_eval(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            assert(g.adj[t->id].size() > 0);

            for (int i = 1; i < g.adj[t->id].size(); ++i) {
                if (get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        != get_output<la::gpu::tensor_like<double>>(get_child(t, i)).vec_size())
                {
                    std::cerr << get_output<la::gpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                        << " != " << get_output<la::gpu::tensor_like<double>>(get_child(t, i)).vec_size() << std::endl;
                    exit(1);
                }
            }

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::tensor_like<double>& m = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
                la::gpu::resize_as(z, m);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& result = get_output<la::gpu::tensor_like<double>>(t);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, i));

                la::gpu::iadd(result, u);
            }
        }

        void add_grad(std::shared_ptr<op_t> t)
        {
            auto& g = *t->graph;

            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            for (int i = 0; i < g.adj[t->id].size(); ++i) {
                auto c = get_child(t, i);

                if (c->grad_needed && c->grad == nullptr) {
                    auto& c_t = get_output<la::gpu::tensor_like<double>>(c);
                    la::gpu::tensor<double> g;
                    la::gpu::resize_as(g, c_t);
                    c->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
                }

                auto& u = get_grad<la::gpu::tensor_like<double>>(c);

                if (c->grad_needed) {
                    la::gpu::iadd(u, grad);
                }
            }
        }

        void sub_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, u);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& result = get_output<la::gpu::tensor_like<double>>(t);
            la::gpu::copy(result, u);
            la::gpu::isub(result, v);
        }

        void sub_grad(std::shared_ptr<op_t> t)
        {
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto u_o = get_child(t, 0);
            auto& u_t = get_output<la::gpu::tensor_like<double>>(u_o);

            if (u_o->grad_needed && u_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u_t);
                u_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto v_o = get_child(t, 1);
            auto& v_t = get_output<la::gpu::tensor_like<double>>(v_o);

            if (v_o->grad_needed && v_o->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, v_t);
                v_o->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& u_grad = get_grad<la::gpu::tensor_like<double>>(u_o);
            auto& v_grad = get_grad<la::gpu::tensor_like<double>>(v_o);

            if (u_o->grad_needed) {
                la::gpu::iadd(u_grad, grad);
            }

            if (v_o->grad_needed) {
                la::gpu::isub(v_grad, grad);
            }
        }

        void logsoftmax_eval(std::shared_ptr<op_t> t)
        {
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::resize_as(z, v);
                t->output = std::make_shared<la::gpu::tensor<double>>(z);
            } else {
                auto& z = get_output<la::gpu::tensor_like<double>>(t);
                la::gpu::zero(z);
            }

            auto& z = get_output<la::gpu::tensor_like<double>>(t);
            op::gpu::logsoftmax(z, v);
        }

        void logsoftmax_grad(std::shared_ptr<op_t> t)
        {
            auto& output = get_output<la::gpu::tensor_like<double>>(t);
            auto& grad = get_grad<la::gpu::tensor_like<double>>(t);

            auto ch = get_child(t, 0);
            auto& ch_t = get_output<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed && ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, ch_t);
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& result = get_grad<la::gpu::tensor_like<double>>(ch);

            if (ch->grad_needed) {
                op::gpu::ilogsoftmax_grad(result, grad, output);
            }
        }

        void resize_as_eval(std::shared_ptr<op_t> t)
        {
            auto c = get_child(t, 0);

            if (t->output == nullptr) {
                auto& c_t = get_output<la::gpu::tensor_like<double>>(c);

                double value = *std::static_pointer_cast<double>(t->data);

                la::gpu::tensor<double> w;
                la::gpu::resize_as(w, c_t, value);

                t->output = std::make_shared<la::gpu::tensor<double>>(w);
            }
        }

        void resize_as_grad(std::shared_ptr<op_t> t)
        {
        }

        struct rep_row_op {
            unsigned int size;
            double *v;

            __host__ __device__
            double operator()(int index)
            {
                return v[index % size];
            }
        };

        void rep_row_to_eval(std::shared_ptr<op_t> t)
        {
            auto c0 = get_child(t, 0);
            auto c1 = get_child(t, 1);

            auto& c0_t = get_output<la::gpu::tensor_like<double>>(c0);
            auto& c1_t = get_output<la::gpu::tensor_like<double>>(c1);

            if (t->output == nullptr) {
                la::gpu::tensor<double> w;
                std::vector<unsigned int> sizes = c0_t.sizes();
                std::vector<unsigned int> new_sizes;
                new_sizes.push_back(c1_t.vec_size() / c0_t.vec_size());
                new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());
                w.resize(new_sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(w);
            }

            auto& w = get_output<la::gpu::tensor_like<double>>(t);

            thrust::transform(thrust::device, thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(int(w.vec_size())),
                thrust::device_ptr<double>(w.data()),
                rep_row_op { c0_t.vec_size(), c0_t.data() });
        }

        void rep_row_to_grad(std::shared_ptr<op_t> t)
        {
            auto c0 = get_child(t, 0);
            auto c1 = get_child(t, 1);

            auto& c0_t = get_output<la::gpu::tensor_like<double>>(c0);

            if (c0->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, c0_t);

                c0->grad = std::make_shared<la::gpu::tensor<double>>(g);
            }

            auto& g = get_grad<la::gpu::tensor_like<double>>(c0);
            auto& v = get_grad<la::gpu::tensor_like<double>>(t);

            la::gpu::weak_tensor<double> z {v.data(), {v.vec_size() / c0_t.vec_size(), g.vec_size()}};

            la::gpu::tensor<double> one;
            one.resize({v.vec_size() / c0_t.vec_size()}, 1);

            la::gpu::mul(g, one, z);
        }

        void dropout_mask_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            double prob;
            std::default_random_engine *gen;
            std::tie(prob, gen) = *std::static_pointer_cast<
                std::tuple<double, std::default_random_engine*>>(t->data);

            la::tensor<double> w;
            w.resize(u.sizes());

            std::bernoulli_distribution bernoulli { 1 - prob };

            double *w_data = w.data();

            for (int i = 0; i < w.vec_size(); ++i) {
                w_data[i] = bernoulli(*gen) / (1 - prob);
            }

            t->output = std::make_shared<la::gpu::tensor<double>>(la::gpu::tensor<double>(w));
        }

        void dropout_mask_grad(std::shared_ptr<op_t> t)
        {
        }

#if 0
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
#endif

    }
}
