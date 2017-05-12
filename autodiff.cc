#include "autodiff/autodiff.h"
#include "la/la-cpu.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>
#include "autodiff/autodiff-op.h"
#include <cblas.h>

namespace autodiff {

    interpreter::interpreter()
        : eval_funcs(autodiff::eval_funcs)
        , grad_funcs(autodiff::grad_funcs)
    {
    }

    interpreter& interpreter::get_instance()
    {
        static interpreter instance;

        return instance;
    }

    op_t::op_t()
        : output(nullptr), grad(nullptr)
    {}

    computation_graph::computation_graph()
        : lazy(false)
    {}

    computation_graph::computation_graph(computation_graph const& graph)
        : lazy(false), vertices(graph.vertices), adj(graph.adj)
    {
        for (auto& v: vertices) {
            v->graph = this;
        }
    }

    computation_graph& computation_graph::operator=(computation_graph const& other)
    {
        lazy = other.lazy;
        vertices = other.vertices;
        adj = other.adj;

        for (auto& v: vertices) {
            v->graph = this;
        }
    }

    std::shared_ptr<op_t> computation_graph::var()
    {
        return make_node("var");
    }

    std::shared_ptr<op_t> computation_graph::make_node(std::string name)
    {
        std::shared_ptr<op_t> result { new op_t };

        result->name = name;
        result->graph = this;
        result->grad_needed = true;
        vertices.push_back(result);
        adj.resize(vertices.size());
        result->id = vertices.size() - 1;

        return result;
    }

    void computation_graph::add_edge(std::shared_ptr<op_t> const& tail,
        std::shared_ptr<op_t> const& head)
    {
        adj[tail->id].push_back(head->id);
    }

    std::shared_ptr<op_t> get_child(std::shared_ptr<op_t> const& t, int index)
    {
        auto& g = *t->graph;

        return g.vertices[g.adj[t->id][index]];
    }

    void add_child(std::shared_ptr<op_t> const& t, std::shared_ptr<op_t> const& ch)
    {
        auto& g = *t->graph;

        g.add_edge(t, ch);
    }

    std::shared_ptr<op_t> weak_var(std::shared_ptr<op_t> t,
        int shift, std::vector<unsigned int> sizes)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("weak_var");

        g.add_edge(result, t);

        result->data = std::make_shared<std::pair<int, std::vector<unsigned int>>>(
            std::make_pair(shift, sizes));

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void weak_var_eval(std::shared_ptr<op_t> t)
    {
        int shift;
        std::vector<unsigned int> sizes;

        std::tie(shift, sizes) = *std::static_pointer_cast<
            std::pair<int, std::vector<unsigned int>>>(t->data);

        auto ch = get_child(t, 0);

        auto& v = get_output<la::cpu::tensor_like<double>>(ch);
        la::cpu::weak_tensor<double> w_v { v.data() + shift, sizes };
        t->output = std::make_shared<la::cpu::weak_tensor<double>>(w_v);

        if (ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            g.resize(v.sizes());
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& g = get_grad<la::cpu::tensor_like<double>>(ch);
        la::cpu::weak_tensor<double> w_g { g.data() + shift, sizes };
        t->grad = std::make_shared<la::cpu::weak_tensor<double>>(w_g);
    }

    void weak_var_grad(std::shared_ptr<op_t> t)
    {
    }

    std::shared_ptr<op_t> mul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("mul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }
    
    void mul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& b = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::cpu::tensor<double> c;

            std::vector<unsigned int> sizes = a.sizes();
            sizes.pop_back();
            sizes.push_back(b.size(b.dim() - 1));

            c.resize(sizes);

            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(c));
        }

        auto& c = get_output<la::cpu::tensor_like<double>>(t);

        la::cpu::mul(c, a, b);
    }

    void mul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::cpu::tensor_like<double>>(a_o);
        auto& b = get_output<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, a);
            a_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, b);
            b_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::cpu::tensor_like<double>>(a_o);
        auto& b_grad = get_grad<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::cpu::rtmul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::cpu::ltmul(b_grad, a, grad);
        }
    }

    std::shared_ptr<op_t> ltmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("ltmul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }
    
    void ltmul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& b = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::cpu::tensor<double> c;
            c.resize({ a.size(a.dim() - 1), b.size(a.dim() - 1) });
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(c));
        }

        auto& c = get_output<la::cpu::tensor_like<double>>(t);
        la::cpu::ltmul(c, a, b);
    }

    void ltmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::cpu::tensor_like<double>>(a_o);
        auto& b = get_output<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, a);
            a_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, b);
            b_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::cpu::tensor_like<double>>(a_o);
        auto& b_grad = get_grad<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::cpu::rtmul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::cpu::mul(b_grad, a, grad);
        }
    }

    std::shared_ptr<op_t> rtmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("rtmul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }
    
    void rtmul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& b = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::cpu::tensor<double> c;
            std::vector<unsigned int> a_sizes = a.sizes();
            std::vector<unsigned int> b_sizes = b.sizes();

            std::vector<unsigned int> sizes;
            sizes.insert(sizes.end(), a_sizes.begin(), a_sizes.end() - 1);
            sizes.insert(sizes.end(), b_sizes.begin(), b_sizes.end() - 1);

            c.resize(sizes);

            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(c));
        }

        auto& c = get_output<la::cpu::tensor_like<double>>(t);
        la::cpu::rtmul(c, a, b);
    }

    void rtmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::cpu::tensor_like<double>>(a_o);
        auto& b = get_output<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, a);
            a_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, b);
            b_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::cpu::tensor_like<double>>(a_o);
        auto& b_grad = get_grad<la::cpu::tensor_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::cpu::mul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::cpu::ltmul(b_grad, grad, a);
        }
    }

    std::shared_ptr<op_t> emul(std::shared_ptr<op_t> s,
        std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("emul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);

        result->output = s->output;
        result->grad = s->grad;

        g.add_edge(s, result);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    std::shared_ptr<op_t> emul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("emul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }
    
    void emul_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, u);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        la::cpu::emul(z, u, v);
    }

    void emul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto u_o = get_child(t, 0);
        auto v_o = get_child(t, 1);

        auto& u = get_output<la::cpu::tensor_like<double>>(u_o);
        auto& v = get_output<la::cpu::tensor_like<double>>(v_o);

        if (u_o->grad_needed && u_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u);
            u_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        if (v_o->grad_needed && v_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, v);
            v_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::cpu::tensor_like<double>>(u_o);
        auto& v_grad = get_grad<la::cpu::tensor_like<double>>(v_o);

        if (u_o->grad_needed) {
            la::cpu::emul(u_grad, grad, v);
        }

        if (v_o->grad_needed) {
            la::cpu::emul(v_grad, grad, u);
        }
    }

    std::shared_ptr<op_t> logistic(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("logistic");
        g.add_edge(result, input);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void logistic_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, v);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        op::logistic(z, v);
    }

    void logistic_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);
        auto& output = get_output<la::cpu::tensor_like<double>>(t);

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            op::ilogistic_grad(result, grad, output);
        }
    }

    std::shared_ptr<op_t> relu(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("relu");
        g.add_edge(result, input);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void relu_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, v);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        op::relu(z, v);
    }

    void relu_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::cpu::tensor_like<double>>(t);
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            op::irelu_grad(result, grad, output);
        }
    }

    std::shared_ptr<op_t> tanh(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("tanh");
        g.add_edge(result, input);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void tanh_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, v);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        op::tanh(z, v);
    }

    void tanh_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);
        auto& output = get_output<la::cpu::tensor_like<double>>(t);

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            op::itanh_grad(result, grad, output);
        }
    }

    std::shared_ptr<op_t> exp(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("exp");
        g.add_edge(result, input);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void exp_eval(std::shared_ptr<op_t> t)
    {
        auto& m = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, m);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);

        double *z_data = z.data();
        double const *m_data = m.data();

        for (int i = 0; i < z.vec_size(); ++i) {
            z_data[i] = std::exp(m_data[i]);
        }
    }

    void exp_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);
        auto& output = get_output<la::cpu::tensor_like<double>>(t);

        assert(grad.vec_size() == output.vec_size());

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            double *result_data = result.data();
            double const *output_data = output.data();
            double const *grad_data = grad.data();

            for (int i = 0; i < result.vec_size(); ++i) {
                result_data[i] += output_data[i] * grad_data[i];
            }
        }
    }

    std::shared_ptr<op_t> add(std::shared_ptr<op_t> t, std::vector<std::shared_ptr<op_t>> ts)
    {
        auto& g = *ts.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("add");

        for (auto& t: ts) {
            g.add_edge(result, t);
        }

        g.add_edge(t, result);

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    std::shared_ptr<op_t> add(std::vector<std::shared_ptr<op_t>> ts)
    {
        auto& g = *ts.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("add");

        for (auto& t: ts) {
            g.add_edge(result, t);
        }

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    std::shared_ptr<op_t> add(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("add");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void add_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        assert(g.adj[t->id].size() > 0);

        for (int i = 1; i < g.adj[t->id].size(); ++i) {
            if (get_output<la::cpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                    != get_output<la::cpu::tensor_like<double>>(get_child(t, i)).vec_size())
            {
                std::cerr << get_output<la::cpu::tensor_like<double>>(get_child(t, i-1)).vec_size()
                    << " != " << get_output<la::cpu::tensor_like<double>>(get_child(t, i)).vec_size() << std::endl;
                exit(1);
            }
        }

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::tensor_like<double>& m = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
            la::cpu::resize_as(z, m);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& result = get_output<la::cpu::tensor_like<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, i));

            la::cpu::iadd(result, u);
        }
    }

    void add_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto c = get_child(t, i);

            if (c->grad_needed && c->grad == nullptr) {
                auto& c_t = get_output<la::cpu::tensor_like<double>>(c);
                la::cpu::tensor<double> g;
                la::cpu::resize_as(g, c_t);
                c->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
            }

            auto& u = get_grad<la::cpu::tensor_like<double>>(c);

            if (c->grad_needed) {
                la::cpu::iadd(u, grad);
            }
        }
    }
    
    std::shared_ptr<op_t> sub(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("sub");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void sub_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, u);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& result = get_output<la::cpu::tensor_like<double>>(t);
        la::cpu::copy(result, u);
        la::cpu::isub(result, v);
    }

    void sub_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto u_o = get_child(t, 0);
        auto& u_t = get_output<la::cpu::tensor_like<double>>(u_o);

        if (u_o->grad_needed && u_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u_t);
            u_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto v_o = get_child(t, 1);
        auto& v_t = get_output<la::cpu::tensor_like<double>>(v_o);

        if (v_o->grad_needed && v_o->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, v_t);
            v_o->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::cpu::tensor_like<double>>(u_o);
        auto& v_grad = get_grad<la::cpu::tensor_like<double>>(v_o);

        if (u_o->grad_needed) {
            la::cpu::iadd(u_grad, grad);
        }

        if (v_o->grad_needed) {
            la::cpu::isub(v_grad, grad);
        }
    }

    std::shared_ptr<op_t> norm(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("norm");
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void norm_eval(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = autodiff::get_output<la::cpu::tensor<double>>(ch);
        t->output = std::make_shared<double>(la::cpu::norm(v));
    }

    void norm_grad(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = autodiff::get_output<la::cpu::tensor<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> z;
            z.resize(v.sizes());
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        double n = autodiff::get_output<double>(t);

        if (ch->grad_needed && n != 0.0) {
            double g = autodiff::get_grad<double>(t);
            auto& z = autodiff::get_grad<la::cpu::tensor<double>>(ch);
            la::cpu::axpy(z, g / n, v);
        }
    }

    std::shared_ptr<op_t> softmax(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("softmax");
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void softmax_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, v);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        op::softmax(z, v);
    }

    void softmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::cpu::tensor_like<double>>(t);
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            op::isoftmax_grad(result, grad, output);
        }
    }
    
    std::shared_ptr<op_t> logsoftmax(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("logsoftmax");
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void logsoftmax_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            la::cpu::resize_as(z, v);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        op::logsoftmax(z, v);
    }

    void logsoftmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::cpu::tensor_like<double>>(t);
        auto& grad = get_grad<la::cpu::tensor_like<double>>(t);

        auto ch = get_child(t, 0);
        auto& ch_t = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, ch_t);
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& result = get_grad<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed) {
            op::ilogsoftmax_grad(result, grad, output);
        }
    }

    std::shared_ptr<op_t> dot(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("dot");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void dot_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        t->output = std::make_shared<double>(la::cpu::dot(v, u));
    }

    void dot_grad(std::shared_ptr<op_t> t)
    {
        auto c0 = get_child(t, 0);
        auto c1 = get_child(t, 1);

        auto& v = get_output<la::cpu::tensor_like<double>>(c0);
        auto& u = get_output<la::cpu::tensor_like<double>>(c1);

        assert(v.vec_size() == u.vec_size());

        double grad = get_grad<double>(t);

        if (c0->grad_needed && c0->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u);
            c0->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& v_grad = get_grad<la::cpu::tensor_like<double>>(c0);

        if (c0->grad_needed) {
            axpy(v_grad, grad, u);
        }

        if (c1->grad_needed && c1->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, v);
            c1->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::cpu::tensor_like<double>>(c1);

        if (c1->grad_needed) {
            axpy(u_grad, grad, v);
        }
    }

    std::shared_ptr<op_t> row_cat(std::vector<std::shared_ptr<op_t>> const& row_vecs)
    {
        auto& g = *row_vecs.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("row_cat");

        for (auto& v: row_vecs) {
            g.add_edge(result, v);
        }

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void row_cat_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;
        assert(g.adj[t->id].size() > 0);

        unsigned int rows = g.adj[t->id].size();

        auto& v0 = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<unsigned int> sizes = v0.sizes();
            std::vector<unsigned int> new_sizes;
            new_sizes.push_back(rows);
            new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());

            la::cpu::tensor<double> z;
            z.resize(new_sizes);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& z = get_output<la::cpu::tensor_like<double>>(t);

        la::cpu::weak_matrix<double> m { z.data(), rows, v0.vec_size() };

        for (int i = 0; i < m.rows(); ++i) {
            auto& vi = get_output<la::cpu::tensor_like<double>>(get_child(t, i));

            assert(vi.vec_size() == m.cols());

            double *vi_data = vi.data();

            for (int j = 0; j < m.cols(); ++j) {
                m(i, j) = vi_data[j];
            }
        }
    }

    void row_cat_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& z = autodiff::get_grad<la::cpu::tensor_like<double>>(t);

        assert(z.size(0) == g.adj[t->id].size());

        la::cpu::weak_matrix<double> m { z.data(), z.size(0), z.vec_size() / z.size(0) };

        for (int i = 0; i < m.rows(); ++i) {
            auto c = get_child(t, i);

            auto& v = autodiff::get_output<la::cpu::tensor_like<double>>(c);

            assert(v.vec_size() == m.cols());

            if (c->grad_needed && c->grad == nullptr) {
                la::cpu::tensor<double> g;
                la::cpu::resize_as(g, v);
                c->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
            }

            auto& g = autodiff::get_grad<la::cpu::tensor_like<double>>(c);

            if (c->grad_needed) {
                double *g_data = g.data();

                for (int j = 0; j < m.cols(); ++j) {
                    g_data[j] += m(i, j);
                }
            }
        }
    }

    std::shared_ptr<op_t> col_cat(std::vector<std::shared_ptr<op_t>> const& col_vecs)
    {
        auto& g = *col_vecs.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("col_cat");

        for (auto& v: col_vecs) {
            g.add_edge(result, v);
        }

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void col_cat_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;
        assert(g.adj[t->id].size() > 0);

        unsigned int cols = g.adj[t->id].size();
        auto v0 = get_child(t, 0);
        unsigned int rows = get_output<la::cpu::vector_like<double>>(v0).size();

        if (t->output == nullptr) {
            la::cpu::tensor<double> m;
            m.resize({ rows, cols });
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(m));
        }

        auto& m = get_output<la::cpu::tensor_like<double>>(t);

        for (int j = 0; j < m.size(1); ++j) {
            auto vj = get_child(t, j);
            auto& v = get_output<la::cpu::tensor_like<double>>(vj);

            assert(v.vec_size() == rows);

            double const *v_data = v.data();

            for (int i = 0; i < m.size(0); ++i) {
                m({ i, j }) = v_data[i];
            }
        }
    }

    void col_cat_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& m = autodiff::get_grad<la::cpu::tensor_like<double>>(t);

        assert(m.size(1) == g.adj[t->id].size());

        for (int j = 0; j < g.adj[t->id].size(); ++j) {
            auto c = get_child(t, j);

            auto& v = autodiff::get_output<la::cpu::tensor_like<double>>(c);

            assert(v.vec_size() == m.size(0));

            if (c->grad_needed && c->grad == nullptr) {
                la::cpu::tensor<double> g;
                la::cpu::resize_as(g, v);
                c->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
            }

            auto& g = autodiff::get_grad<la::cpu::tensor_like<double>>(c);

            if (c->grad_needed) {
                double *g_data = g.data();

                for (int i = 0; i < m.size(0); ++i) {
                    g_data[i] += m({ i, j });
                }
            }
        }
    }

    std::shared_ptr<op_t> row_at(std::shared_ptr<op_t> const& t, int i)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("row_at");
        result->data = std::make_shared<unsigned int>(i);
    
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void row_at_eval(std::shared_ptr<op_t> t)
    {
        int i = *std::static_pointer_cast<unsigned int>(t->data);
        la::cpu::weak_matrix<double> m = get_output<la::cpu::tensor_like<double>>(get_child(t, 0)).as_matrix();
        assert(i < m.rows());
        la::cpu::weak_tensor<double> v { m.data() + i * m.cols(), { m.cols() } };
        t->output = std::make_shared<la::cpu::weak_tensor<double>>(v);
    }

    void row_at_grad(std::shared_ptr<op_t> t)
    {
        int i = *std::static_pointer_cast<unsigned int>(t->data);

        auto c = get_child(t, 0);
        auto& m = get_output<la::cpu::tensor_like<double>>(c);

        if (c->grad_needed && c->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, m);
            c->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& v = get_grad<la::cpu::tensor_like<double>>(t);
        auto& g = get_grad<la::cpu::tensor_like<double>>(c);

        if (c->grad_needed) {
            la::cpu::weak_matrix<double> mat = m.as_matrix();

            double *g_data = g.data();
            double const *v_data = v.data();

            unsigned int cols = mat.cols();

            for (int d = 0; d < v.vec_size(); ++d) {
                g_data[i * cols + d] += v_data[d];
            }
        }
    }

    std::shared_ptr<op_t> reshape(std::shared_ptr<op_t> const& t, std::vector<unsigned int> sizes)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("reshape");
        result->data = std::make_shared<std::vector<unsigned int>>(sizes);
    
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void reshape_eval(std::shared_ptr<op_t> t)
    {
        std::vector<unsigned int>& sizes = *std::static_pointer_cast<std::vector<unsigned int>>(t->data);
        la::cpu::tensor_like<double>& input = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        unsigned int d = (sizes.size() == 0 ? 0 : 1);
        for (int i = 0; i < sizes.size(); ++i) {
            d *= sizes[i];
        }
        assert(d <= input.vec_size());

        la::cpu::weak_tensor<double> result { input.data(), sizes };
        t->output = std::make_shared<la::cpu::weak_tensor<double>>(result);
    }

    void reshape_grad(std::shared_ptr<op_t> t)
    {
        std::vector<unsigned int>& sizes = *std::static_pointer_cast<std::vector<unsigned int>>(t->data);

        auto c = get_child(t, 0);
        auto& input = get_output<la::cpu::tensor_like<double>>(c);

        if (c->grad_needed && c->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, input);
            c->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& output_grad = get_grad<la::cpu::tensor_like<double>>(t);
        auto& input_grad = get_grad<la::cpu::tensor_like<double>>(c);

        if (c->grad_needed) {
            double const *output_grad_data = output_grad.data();
            double *input_grad_data = input_grad.data();

            for (int i = 0; i < output_grad.vec_size(); ++i) {
                input_grad_data[i] += output_grad_data[i];
            }
        }
    }

    std::shared_ptr<op_t> resize_as(std::shared_ptr<op_t> const& t, double value)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("resize_as");
    
        result->data = std::make_shared<double>(value);

        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void resize_as_eval(std::shared_ptr<op_t> t)
    {
        auto c = get_child(t, 0);

        if (t->output == nullptr) {
            auto& c_t = get_output<la::cpu::tensor_like<double>>(c);

            double value = *std::static_pointer_cast<double>(t->data);

            la::cpu::tensor<double> w;
            la::cpu::resize_as(w, c_t, value);

            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(w));
        }
    }

    void resize_as_grad(std::shared_ptr<op_t> t)
    {
    }

    std::shared_ptr<op_t> rep_row_to(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("rep_row_to");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void rep_row_to_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        assert(v.vec_size() % u.vec_size() == 0);

        if (t->output == nullptr) {
            la::cpu::tensor<double> w;
            std::vector<unsigned int> sizes = u.sizes();
            std::vector<unsigned int> new_sizes;
            new_sizes.push_back(v.vec_size() / u.vec_size());
            new_sizes.insert(new_sizes.end(), sizes.begin(), sizes.end());
            w.resize(new_sizes);

            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(w));
        }

        auto& w = get_output<la::cpu::tensor_like<double>>(t);

        la::cpu::weak_matrix<double> w_mat {w.data(), v.vec_size() / u.vec_size(), u.vec_size()};

        la::cpu::vector<double> one;
        one.resize(v.vec_size() / u.vec_size(), 1);

        la::cpu::outer_prod(w_mat, one, u.as_vector());
    }

    void rep_row_to_grad(std::shared_ptr<op_t> t)
    {
        auto u_op = get_child(t, 0);

        auto& u = get_output<la::cpu::tensor_like<double>>(u_op);
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        if (u_op->grad_needed && u_op->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u);
            u_op->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& g_w = get_grad<la::cpu::tensor_like<double>>(t);
        auto& g_u = get_grad<la::cpu::tensor_like<double>>(u_op);

        if (u_op->grad_needed) {
            la::cpu::weak_matrix<double> z {g_w.data(), g_w.vec_size() / u.vec_size(), u.vec_size()};

            la::cpu::vector<double> one;
            one.resize({g_w.vec_size() / u.vec_size()}, 1);

            la::cpu::lmul(g_u.as_vector(), one, z);
        }
    }

    std::shared_ptr<op_t> rep_col_to(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("rep_col_to");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void rep_col_to_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        assert(v.vec_size() % u.vec_size() == 0);

        if (t->output == nullptr) {
            la::cpu::tensor<double> w;
            std::vector<unsigned int> sizes = u.sizes();
            sizes.push_back(v.vec_size() / u.vec_size());
            w.resize(sizes);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(w));
        }

        auto& w = get_output<la::cpu::tensor_like<double>>(t);

        la::cpu::vector<double> one;
        one.resize(v.vec_size() / u.vec_size(), 1);

        la::cpu::outer_prod(w.as_matrix(), u.as_vector(), one);
    }

    void rep_col_to_grad(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        assert(v.vec_size() % u.vec_size() == 0);

        auto u_op = get_child(t, 0);

        if (u_op->grad_needed && u_op->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u);
            u_op->grad = std::make_shared<la::cpu::tensor<double>>(g);
        }

        auto& g_w = get_grad<la::cpu::tensor_like<double>>(t);
        auto& g_u = get_grad<la::cpu::tensor_like<double>>(u_op);

        if (u_op->grad_needed) {
            la::cpu::vector<double> one;
            one.resize(v.vec_size() / u.vec_size(), 1);
            
            la::cpu::mul(g_u.as_vector(), g_w.as_matrix(), one);
        }
    }

    std::shared_ptr<op_t> corr_linearize(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("corr_linearize");

        g.add_edge(result, t1);
        g.add_edge(result, t2);

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void corr_linearize_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        la::cpu::weak_matrix<double> v_mat = v.as_matrix();

        if (t->output == nullptr) {
            la::cpu::tensor<double> w;
            w.resize(std::vector<unsigned int> { u.size(0), u.size(1), v_mat.rows() });
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(w));
        }

        la::cpu::tensor_like<double>& w = get_output<la::cpu::tensor_like<double>>(t);
        la::cpu::corr_linearize(w, u, v.size(0), v.size(1));
    }

    void corr_linearize_grad(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));
        auto& v = get_output<la::cpu::tensor_like<double>>(get_child(t, 1));

        la::cpu::weak_matrix<double> v_mat = v.as_matrix();

        auto u_op = get_child(t, 0);

        if (u_op->grad_needed && u_op->grad == nullptr) {
            la::cpu::tensor<double> g;
            la::cpu::resize_as(g, u);
            u_op->grad = std::make_shared<la::cpu::tensor<double>>(std::move(g));
        }

        auto& g_w = get_grad<la::cpu::tensor_like<double>>(t);
        auto& g_u = get_grad<la::cpu::tensor_like<double>>(u_op);

        if (u_op->grad_needed) {
            op::corr_linearize_grad(g_u, g_w, v.size(0), v.size(1));
        }
    }

    std::shared_ptr<op_t> corr(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2)
    {
        auto t = corr_linearize(t1, t2);
        return mul(t, t2);
    }

    std::shared_ptr<op_t> seg_max(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;
    
        std::shared_ptr<op_t> result = g.make_node("seg-max");
    
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }
    
        return result;
    }
    
    void seg_max_eval(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = get_output<la::cpu::tensor_like<double>>(ch);
    
        if (t->output == nullptr) {
            double inf = std::numeric_limits<double>::infinity();
            la::cpu::tensor<double> z;
            z.resize({v.size(2)}, -inf);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }
    
        auto& z = get_output<la::cpu::tensor_like<double>>(t);
        std::vector<std::pair<int, int>> argmax;
        argmax.resize(z.size(0));
    
        for (int i = 0; i < v.size(0); ++i) {
            for (int j = 0; j < v.size(1); ++j) {
                for (int c = 0; c < v.size(2); ++c) {
                    if (v({i, j, c}) > z({c})) {
                        argmax[c] = std::make_pair(i, j);
                        z({c}) = v({i, j, c});
                    }
                }
            }
        }
    
        t->data = std::make_shared<std::vector<std::pair<int, int>>>(argmax);
    }
    
    void seg_max_grad(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> z;
            z.resize(v.sizes());
            ch->grad = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        auto& g = get_grad<la::cpu::tensor_like<double>>(t);
        auto& z = get_grad<la::cpu::tensor_like<double>>(ch);

        auto& argmin = *std::static_pointer_cast<std::vector<std::pair<int, int>>>(t->data);

        if (ch->grad_needed) {
            for (int c = 0; c < argmin.size(); ++c) {
                z({argmin[c].first, argmin[c].second, c}) += g({c});
            }
        }
    }

    std::shared_ptr<op_t> high_pass_k(std::shared_ptr<op_t> t, int k)
    {
        auto& g = *t->graph;
    
        std::shared_ptr<op_t> result = g.make_node("high-pass-k");

        result->data = std::make_shared<std::pair<int, std::vector<int>>>(
            std::make_pair(k, std::vector<int>()));
    
        g.add_edge(result, t);
    
        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }
    
        return result;
    }

    void high_pass_k_eval(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = get_output<la::cpu::tensor_like<double>>(ch);

        int k = std::static_pointer_cast<std::pair<int, std::vector<int>>>(t->data)->first;

        if (t->output == nullptr) {
            la::cpu::tensor<double> z;
            z.resize(v.sizes());
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(z));
        }

        std::vector<std::pair<double, int>> heap;
        for (int i = 0; i < v.vec_size(); ++i) {
            heap.push_back(std::make_pair(v.data()[i], i));
        }

        auto less = [](std::pair<double, int> const& p1, std::pair<double, int> const& p2) {
            return p1.first < p2.first;
        };

        std::vector<int> indices;

        std::make_heap(heap.begin(), heap.end(), less);

        auto& z = get_output<la::cpu::tensor<double>>(t);

        for (int i = 0; i < k; ++i) {
            std::pop_heap(heap.begin(), heap.end(), less);
            auto& p = heap.back();
            z({p.second}) = p.first;
            indices.push_back(p.second);
            heap.pop_back();
        }

        std::static_pointer_cast<std::pair<int, std::vector<int>>>(t->data)->second = indices;
    }

    void high_pass_k_grad(std::shared_ptr<op_t> t)
    {
        auto ch = get_child(t, 0);
        auto& v = get_output<la::cpu::tensor_like<double>>(ch);

        if (ch->grad_needed && ch->grad == nullptr) {
            la::cpu::tensor<double> z;
            z.resize(v.sizes());
            ch->grad = std::make_shared<la::cpu::tensor<double>>(z);
        }

        auto& g = get_grad<la::cpu::tensor_like<double>>(t);
        auto& z = get_grad<la::cpu::tensor_like<double>>(ch);

        std::vector<int>& indices = std::static_pointer_cast<std::pair<int, std::vector<int>>>(t->data)->second;

        if (ch->grad_needed) {
            for (auto& i: indices) {
                z.data()[i] += g.data()[i];
            }
        }
    }

    std::shared_ptr<op_t> dropout_mask(std::shared_ptr<op_t> t, double prob, std::default_random_engine& gen)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("dropout_mask");

        result->data = std::make_shared<std::tuple<double, std::default_random_engine*>>(
            std::make_tuple(prob, &gen));

        g.add_edge(result, t);

        if (!g.lazy) {
            eval_vertex(result, autodiff::interpreter::get_instance().eval_funcs);
        }

        return result;
    }

    void dropout_mask_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::cpu::tensor_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::cpu::tensor<double> w;
            la::cpu::resize_as(w, u);
            t->output = std::make_shared<la::cpu::tensor<double>>(std::move(w));
        }

        double prob;
        std::default_random_engine *gen;
        std::tie(prob, gen) = *std::static_pointer_cast<
            std::tuple<double, std::default_random_engine*>>(t->data);

        auto& w = get_output<la::cpu::tensor_like<double>>(t);

        std::bernoulli_distribution bernoulli { 1 - prob };

        double *w_data = w.data();

        for (int i = 0; i < w.vec_size(); ++i) {
            w_data[i] = bernoulli(*gen) / (1 - prob);
        }
    }

    void dropout_mask_grad(std::shared_ptr<op_t> t)
    {
    }

    std::vector<std::shared_ptr<op_t>> natural_topo_order(computation_graph const& graph)
    {
        std::vector<std::shared_ptr<op_t>> result;
        for (int i = graph.vertices.size() - 1; i >= 0; --i) {
            result.push_back(graph.vertices.at(i));
        }
        return result;
    }

    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots,
        std::vector<std::shared_ptr<op_t>> const& boundaries)
    {
        enum class action_t {
            color_grey,
            color_black
        };

        enum class color_t {
            white,
            grey,
            black
        };

        std::unordered_set<std::shared_ptr<op_t>> boundary_set
            { boundaries.begin(), boundaries.end() };

        auto& g = *roots.front()->graph;

        std::vector<std::shared_ptr<op_t>> order;

        std::vector<color_t> color;
        color.resize(g.vertices.size(), color_t::white);

        std::vector<std::pair<action_t, std::shared_ptr<op_t>>> stack;

        for (auto& r: roots) {
            stack.push_back(std::make_pair(action_t::color_grey, r));
        }

        while (stack.size() != 0) {
            action_t a;
            std::shared_ptr<op_t> t;

            std::tie(a, t) = stack.back();

            stack.pop_back();

            if (a == action_t::color_grey) {
                if (color[t->id] != color_t::white) {
                    continue;
                }

                color[t->id] = color_t::grey;

                stack.push_back(std::make_pair(action_t::color_black, t));

                for (int i = 0; i < g.adj[t->id].size(); ++i) {
                    auto c = get_child(t, i);

                    if (ebt::in(c, boundary_set)) {
                        continue;
                    }

                    if (color[c->id] == color_t::white) {
                        stack.push_back(std::make_pair(action_t::color_grey, c));
                    }
                }
            } else if (a == action_t::color_black) {
                if (color[t->id] != color_t::grey) {
                    std::cerr << "invalid color" << std::endl;
                    exit(1);
                }

                color[t->id] = color_t::black;
                order.push_back(t);
            } else {
                std::cerr << "unknown action" << std::endl; 
                exit(1);
            }
        }

        std::reverse(order.begin(), order.end());

        return order;
    }

    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots)
    {
        return topo_order(roots, std::vector<std::shared_ptr<autodiff::op_t>>{});
    }

    std::vector<std::shared_ptr<op_t>> topo_order(std::shared_ptr<op_t> const& root)
    {
        return topo_order(std::vector<std::shared_ptr<op_t>> { root });
    }

    void eval_vertex(std::shared_ptr<op_t> const& t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        auto& g = *t->graph;

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            if (g.vertices[g.adj[t->id][i]]->grad_needed) {
                // TODO: dangerous?
                t->grad_needed = true;
                break;
            }
        }

        funcs.at(t->name)(t);
    }

    void eval(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        for (int i = topo_order.size() - 1; i >= 0; --i) {
            eval_vertex(topo_order[i], funcs);
        }
    }

    void eval(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        auto order = topo_order(root);
        eval(order, funcs);
    }

    void grad(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        for (int i = 0; i < topo_order.size(); ++i) {
            if (topo_order[i]->grad_needed) {
                if (topo_order[i]->grad == nullptr) {
                    for (int j = 0; j <= i; ++j) {
                        std::cout << topo_order[j]->name << std::endl;
                    }

                    throw std::logic_error { "no grad" };
                }

                eval_vertex(topo_order[i], funcs);
            }
        }
    }

    void grad(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        auto order = topo_order(root);
        grad(order, funcs);
    }

    void guarded_grad(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        for (int i = 0; i < topo_order.size(); ++i) {
            if (topo_order[i]->grad_needed && topo_order[i]->grad != nullptr) {
                eval_vertex(topo_order[i], funcs);
            }
        }
    }

    void guarded_grad(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
        auto order = topo_order(root);
        guarded_grad(order, funcs);
    }

    void clear_grad(std::vector<std::shared_ptr<op_t>> const& topo_order)
    {
        for (auto& t: topo_order) {
            t->grad = nullptr;
        }
    }

    void clear_grad(std::shared_ptr<op_t> const& root)
    {
        auto order = topo_order(root);
        clear_grad(order);
    }

}
