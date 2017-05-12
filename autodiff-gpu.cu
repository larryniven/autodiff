#include "autodiff/autodiff-gpu.h"
#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

namespace autodiff {

    namespace gpu {

        void weak_var_eval(std::shared_ptr<op_t> t)
        {
            int shift;
            std::vector<unsigned int> sizes;

            std::tie(shift, sizes) = *std::static_pointer_cast<
                std::pair<int, std::vector<unsigned int>>>(t->data);

            auto ch = get_child(t, 0);

            auto& v = get_output<la::gpu::tensor_like<double>>(ch);
            la::gpu::weak_tensor<double> w_v { v.data() + shift, sizes };
            t->output = std::make_shared<la::gpu::weak_tensor<double>>(w_v);

            if (ch->grad == nullptr) {
                la::gpu::tensor<double> g;
                g.resize(v.sizes());
                ch->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& g = get_grad<la::gpu::tensor_like<double>>(ch);
            la::gpu::weak_tensor<double> w_g { g.data() + shift, sizes };
            t->grad = std::make_shared<la::gpu::weak_tensor<double>>(w_g);
        }

        void weak_var_grad(std::shared_ptr<op_t> t)
        {
        }

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

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(c));
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
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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
                        << " != " << get_output<la::gpu::tensor_like<double>>(
                            get_child(t, i)).vec_size() << std::endl;
                    exit(1);
                }
            }

            if (t->output == nullptr) {
                la::gpu::tensor<double> z;
                la::gpu::tensor_like<double>& m = get_output<la::gpu::tensor_like<double>>(
                    get_child(t, 0));
                la::gpu::resize_as(z, m);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(z));
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

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }
        }

        void resize_as_grad(std::shared_ptr<op_t> t)
        {
        }

        void rep_row_to_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            if (t->output == nullptr) {
                la::gpu::tensor<double> w;
                std::vector<unsigned int> sizes = u.sizes();
                std::vector<unsigned int> new_sizes;
                new_sizes.push_back(v.vec_size() / u.vec_size());
                new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());
                w.resize(new_sizes);

                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }

            auto& w = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::weak_matrix<double> w_mat {w.data(), v.vec_size() / u.vec_size(), u.vec_size()};

            la::gpu::vector<double> one;
            one.resize(v.vec_size() / u.vec_size(), 1);

            la::gpu::outer_prod(w_mat, one, u.as_vector());
        }

        void rep_row_to_grad(std::shared_ptr<op_t> t)
        {
            auto u_op = get_child(t, 0);

            auto& u = get_output<la::gpu::tensor_like<double>>(u_op);
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            if (u_op->grad_needed && u_op->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_op->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& g_w = get_grad<la::gpu::tensor_like<double>>(t);
            auto& g_u = get_grad<la::gpu::tensor_like<double>>(u_op);

            if (u_op->grad_needed) {
                la::gpu::weak_matrix<double> z {
                    g_w.data(), g_w.vec_size() / u.vec_size(), u.vec_size()};

                la::gpu::vector<double> one;
                one.resize({g_w.vec_size() / u.vec_size()}, 1);

                la::gpu::lmul(g_u.as_vector(), one, z);
            }
        }

        void rep_col_to_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            if (t->output == nullptr) {
                la::gpu::tensor<double> w;
                std::vector<unsigned int> sizes = u.sizes();
                sizes.push_back(v.vec_size() / u.vec_size());
                w.resize(sizes);
                t->output = std::make_shared<la::gpu::tensor<double>>(std::move(w));
            }

            auto& w = get_output<la::gpu::tensor_like<double>>(t);

            la::gpu::vector<double> one;
            one.resize(v.vec_size() / u.vec_size(), 1);

            la::gpu::outer_prod(w.as_matrix(), u.as_vector(), one);
        }

        void rep_col_to_grad(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));
            auto& v = get_output<la::gpu::tensor_like<double>>(get_child(t, 1));

            assert(v.vec_size() % u.vec_size() == 0);

            auto u_op = get_child(t, 0);

            if (u_op->grad_needed && u_op->grad == nullptr) {
                la::gpu::tensor<double> g;
                la::gpu::resize_as(g, u);
                u_op->grad = std::make_shared<la::gpu::tensor<double>>(std::move(g));
            }

            auto& g_w = get_grad<la::gpu::tensor_like<double>>(t);
            auto& g_u = get_grad<la::gpu::tensor_like<double>>(u_op);

            if (u_op->grad_needed) {
                la::gpu::vector<double> one;
                one.resize(v.vec_size() / u.vec_size(), 1);
                
                la::gpu::mul(g_u.as_vector(), g_w.as_matrix(), one);
            }
        }

        void dropout_mask_eval(std::shared_ptr<op_t> t)
        {
            auto& u = get_output<la::gpu::tensor_like<double>>(get_child(t, 0));

            double prob;
            std::default_random_engine *gen;
            std::tie(prob, gen) = *std::static_pointer_cast<
                std::tuple<double, std::default_random_engine*>>(t->data);

            la::cpu::tensor<double> w;
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

    }
}
