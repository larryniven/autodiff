#include "autodiff/autodiff.h"
#include "la/la.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>
#include "autodiff/autodiff-op.h"
#include <cblas.h>

namespace autodiff {

    op_t::op_t()
        : output(nullptr), grad(nullptr)
    {}

    computation_graph::computation_graph()
    {}

    computation_graph::computation_graph(computation_graph const& graph)
        : vertices(graph.vertices), adj(graph.adj)
    {
        for (auto& v: vertices) {
            v->graph = this;
        }
    }

    computation_graph& computation_graph::operator=(computation_graph const& other)
    {
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

    std::shared_ptr<op_t> mul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("mul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void mul_eval(std::shared_ptr<op_t> t)
    {
        auto& A = get_output<la::matrix_like<double>>(get_child(t, 0));
        auto& v = get_output<la::vector_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::vector<double> u;
            u.resize(A.rows());
            t->output = std::make_shared<la::vector<double>>(u);
        } else {
            auto& u = get_output<la::vector_like<double>>(t);
            la::zero(u);
        }

        auto& u = get_output<la::vector_like<double>>(t);
        la::mul(u, A, v);
    }

    void mul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto A_o = get_child(t, 0);
        auto v_o = get_child(t, 1);

        auto& A = get_output<la::matrix_like<double>>(A_o);
        auto& v = get_output<la::vector_like<double>>(v_o);

        if (A_o->grad_needed && A_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(grad.size(), v.size());
            A_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        if (v_o->grad_needed && v_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(A.cols());
            v_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& A_grad = get_grad<la::matrix_like<double>>(A_o);
        auto& v_grad = get_grad<la::vector_like<double>>(v_o);

        if (A_o->grad_needed) {
            la::outer_prod(A_grad, grad, v);
        }

        if (v_o->grad_needed) {
            la::lmul(v_grad, grad, A);
        }
    }

    std::shared_ptr<op_t> mmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("mmul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void mmul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::matrix_like<double>>(get_child(t, 0));
        auto& b = get_output<la::matrix_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::matrix<double> c;
            c.resize(a.rows(), b.cols());
            t->output = std::make_shared<la::matrix<double>>(c);
        } else {
            auto& c = get_output<la::matrix_like<double>>(t);
            la::zero(c);
        }

        auto& c = get_output<la::matrix_like<double>>(t);
        la::mul(c, a, b);
    }

    void mmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::matrix_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::matrix_like<double>>(a_o);
        auto& b = get_output<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(a.rows(), a.cols());
            a_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(b.rows(), b.cols());
            b_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::matrix_like<double>>(a_o);
        auto& b_grad = get_grad<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::rtmul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::ltmul(b_grad, a, grad);
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
    
        return result;
    }
    
    void ltmul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::matrix_like<double>>(get_child(t, 0));
        auto& b = get_output<la::matrix_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::matrix<double> c;
            c.resize(a.cols(), b.cols());
            t->output = std::make_shared<la::matrix<double>>(c);
        } else {
            auto& c = get_output<la::matrix_like<double>>(t);
            la::zero(c);
        }

        auto& c = get_output<la::matrix_like<double>>(t);
        la::ltmul(c, a, b);
    }

    void ltmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::matrix_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::matrix_like<double>>(a_o);
        auto& b = get_output<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(a.rows(), a.cols());
            a_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(b.rows(), b.cols());
            b_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::matrix_like<double>>(a_o);
        auto& b_grad = get_grad<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::rtmul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::mul(b_grad, a, grad);
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
    
        return result;
    }
    
    void rtmul_eval(std::shared_ptr<op_t> t)
    {
        auto& a = get_output<la::matrix_like<double>>(get_child(t, 0));
        auto& b = get_output<la::matrix_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::matrix<double> c;
            c.resize(a.rows(), b.rows());
            t->output = std::make_shared<la::matrix<double>>(c);
        } else {
            auto& c = get_output<la::matrix_like<double>>(t);
            la::zero(c);
        }

        auto& c = get_output<la::matrix_like<double>>(t);
        la::rtmul(c, a, b);
    }

    void rtmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::matrix_like<double>>(t);

        auto a_o = get_child(t, 0);
        auto b_o = get_child(t, 1);

        auto& a = get_output<la::matrix_like<double>>(a_o);
        auto& b = get_output<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed && a_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(a.rows(), a.cols());
            a_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        if (b_o->grad_needed && b_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(b.rows(), b.cols());
            b_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        auto& a_grad = get_grad<la::matrix_like<double>>(a_o);
        auto& b_grad = get_grad<la::matrix_like<double>>(b_o);

        if (a_o->grad_needed) {
            la::mul(a_grad, grad, b);
        }

        if (b_o->grad_needed) {
            la::ltmul(b_grad, grad, a);
        }
    }

    std::shared_ptr<op_t> lmul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("lmul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void lmul_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));
        auto& A = get_output<la::matrix_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::vector<double> u;
            u.resize(A.cols());
            t->output = std::make_shared<la::vector<double>>(u);
        } else {
            auto& u = get_output<la::vector_like<double>>(t);
            la::zero(u);
        }

        auto& u = get_output<la::vector_like<double>>(t);
        la::lmul(u, v, A);
    }

    void lmul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto v_o = get_child(t, 0);
        auto A_o = get_child(t, 1);

        auto& v = get_output<la::vector_like<double>>(v_o);
        auto& A = get_output<la::matrix_like<double>>(A_o);

        if (A_o->grad_needed && A_o->grad == nullptr) {
            la::matrix<double> g;
            g.resize(v.size(), grad.size());
            A_o->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        if (v_o->grad_needed && v_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(A.rows());
            v_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& v_grad = get_grad<la::vector_like<double>>(v_o);
        auto& A_grad = get_grad<la::matrix_like<double>>(A_o);

        if (A_o->grad_needed) {
            la::outer_prod(A_grad, v, grad);
        }

        if (v_o->grad_needed) {
            la::mul(v_grad, A, grad);
        }
    }

    std::shared_ptr<op_t> emul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("emul");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void emul_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::vector_like<double>>(get_child(t, 0));
        auto& v = get_output<la::vector_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(u.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        la::emul(z, u, v);
    }

    void emul_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto u_o = get_child(t, 0);
        auto v_o = get_child(t, 1);

        auto& u = get_output<la::vector_like<double>>(u_o);
        auto& v = get_output<la::vector_like<double>>(v_o);

        if (u_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(u.size());
            u_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        if (v_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(v.size());
            v_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::vector_like<double>>(u_o);
        auto& v_grad = get_grad<la::vector_like<double>>(v_o);

        la::emul(u_grad, grad, v);
        la::emul(v_grad, grad, u);
    }

    std::shared_ptr<op_t> logistic(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("logistic");
        g.add_edge(result, input);
    
        return result;
    }

    void logistic_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(v.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        op::logistic(z, v);
    }

    void logistic_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);
        auto& output = get_output<la::vector_like<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            la::vector<double> g;
            g.resize(output.size());
            ch->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& result = get_grad<la::vector_like<double>>(ch);
        op::ilogistic_grad(result, grad, output);
    }

    std::shared_ptr<op_t> relu(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("relu");
        g.add_edge(result, input);
    
        return result;
    }

    void relu_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(v.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        op::relu(z, v);
    }

    void relu_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector_like<double>>(t);
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            la::vector<double> g;
            g.resize(output.size());
            ch->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& result = get_grad<la::vector_like<double>>(ch);
        op::irelu_grad(result, grad, output);
    }

    std::shared_ptr<op_t> tanh(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("tanh");
        g.add_edge(result, input);
    
        return result;
    }

    void tanh_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(v.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        op::tanh(z, v);
    }

    void tanh_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);
        auto& output = get_output<la::vector_like<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            la::vector<double> g;
            g.resize(output.size());
            ch->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& result = get_grad<la::vector_like<double>>(ch);
        op::itanh_grad(result, grad, output);
    }

    std::shared_ptr<op_t> mexp(std::shared_ptr<op_t> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op_t> result = g.make_node("mexp");
        g.add_edge(result, input);
    
        return result;
    }

    void mexp_eval(std::shared_ptr<op_t> t)
    {
        auto& m = get_output<la::matrix_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::matrix<double> z;
            z.resize(m.rows(), m.cols());
            t->output = std::make_shared<la::matrix<double>>(std::move(z));
        } else {
            auto& z = get_output<la::matrix_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::matrix_like<double>>(t);

        for (int i = 0; i < z.rows(); ++i) {
            for (int j = 0; j < z.cols(); ++j) {
                z(i, j) = std::exp(m(i, j));
            }
        }
    }

    void mexp_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::matrix_like<double>>(t);
        auto& output = get_output<la::matrix_like<double>>(t);

        assert(grad.rows() == output.rows() && grad.cols() == output.cols());

        auto ch = get_child(t, 0);

        if (ch->grad == nullptr) {
            la::matrix<double> g;
            g.resize(output.rows(), output.cols());
            ch->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        auto& result = get_grad<la::matrix_like<double>>(ch);

        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                result(i, j) += output(i, j) * grad(i, j);
            }
        }
    }

    std::shared_ptr<op_t> add(std::vector<std::shared_ptr<op_t>> ts)
    {
        auto& g = *ts.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("add");

        for (auto& t: ts) {
            g.add_edge(result, t);
        }

        return result;
    }

    std::shared_ptr<op_t> add(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("add");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void add_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        assert(g.adj[t->id].size() > 0);

#ifndef NDEBUG
        for (int i = 1; i < g.adj[t->id].size(); ++i) {
            assert(get_output<la::vector_like<double>>(get_child(t, i-1)).size()
                == get_output<la::vector_like<double>>(get_child(t, i)).size());
        }
#endif

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(get_output<la::vector<double>>(get_child(t, 0)).size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& result = get_output<la::vector_like<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto& u = get_output<la::vector_like<double>>(get_child(t, i));

            la::iadd(result, u);
        }
    }

    void add_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& grad = get_grad<la::vector_like<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto c = get_child(t, i);

            if (c->grad == nullptr) {
                la::vector<double> g;
                g.resize(grad.size());
                c->grad = std::make_shared<la::vector<double>>(std::move(g));
            }

            auto& u = get_grad<la::vector_like<double>>(c);

            la::iadd(u, grad);
        }
    }
    
    std::shared_ptr<op_t> sub(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("sub");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void sub_eval(std::shared_ptr<op_t> t)
    {
        auto& u = get_output<la::vector_like<double>>(get_child(t, 0));
        auto& v = get_output<la::vector_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(u.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& result = get_output<la::vector_like<double>>(t);
        la::copy(result, u);
        la::isub(result, v);
    }

    void sub_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto u_o = get_child(t, 0);
        if (u_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(grad.size());
            u_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto v_o = get_child(t, 1);
        if (v_o->grad == nullptr) {
            la::vector<double> g;
            g.resize(grad.size());
            v_o->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::vector_like<double>>(u_o);
        auto& v_grad = get_grad<la::vector_like<double>>(v_o);

        la::iadd(u_grad, grad);
        la::isub(v_grad, grad);
    }
    
    std::shared_ptr<op_t> softmax(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("softmax");
        g.add_edge(result, t);
    
        return result;
    }

    void softmax_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(v.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        op::softmax(z, v);
    }

    void softmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector_like<double>>(t);
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            la::vector<double> g;
            g.resize(output.size());
            ch->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& result = get_grad<la::vector_like<double>>(ch);
        op::isoftmax_grad(result, grad, output);
    }
    
    std::shared_ptr<op_t> logsoftmax(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("logsoftmax");
        g.add_edge(result, t);
    
        return result;
    }

    void logsoftmax_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            la::vector<double> z;
            z.resize(v.size());
            t->output = std::make_shared<la::vector<double>>(z);
        } else {
            auto& z = get_output<la::vector_like<double>>(t);
            la::zero(z);
        }

        auto& z = get_output<la::vector_like<double>>(t);
        op::logsoftmax(z, v);
    }

    void logsoftmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector_like<double>>(t);
        auto& grad = get_grad<la::vector_like<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            la::vector<double> g;
            g.resize(output.size());
            ch->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& result = get_grad<la::vector_like<double>>(ch);
        op::ilogsoftmax_grad(result, grad, output);
    }
    
    std::shared_ptr<op_t> dot(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("dot");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void dot_eval(std::shared_ptr<op_t> t)
    {
        auto& v = get_output<la::vector_like<double>>(get_child(t, 0));
        auto& u = get_output<la::vector_like<double>>(get_child(t, 1));

        t->output = std::make_shared<double>(la::dot(v, u));
    }

    void dot_grad(std::shared_ptr<op_t> t)
    {
        auto c0 = get_child(t, 0);
        auto c1 = get_child(t, 1);

        auto& v = get_output<la::vector_like<double>>(c0);
        auto& u = get_output<la::vector_like<double>>(c1);

        assert(v.size() == u.size());

        double grad = get_grad<double>(t);

        if (c0->grad == nullptr) {
            la::vector<double> g;
            g.resize(u.size());
            c0->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& v_grad = get_grad<la::vector_like<double>>(c0);

        cblas_daxpy(v_grad.size(), grad, u.data(), 1, v_grad.data(), 1);

        if (c1->grad == nullptr) {
            la::vector<double> g;
            g.resize(v.size());
            c1->grad = std::make_shared<la::vector<double>>(std::move(g));
        }

        auto& u_grad = get_grad<la::vector_like<double>>(c1);

        cblas_daxpy(u_grad.size(), grad, v.data(), 1, u_grad.data(), 1);
    }

    std::shared_ptr<op_t> row_cat(std::vector<std::shared_ptr<op_t>> const& row_vecs)
    {
        auto& g = *row_vecs.front()->graph;

        std::shared_ptr<op_t> result = g.make_node("row_cat");

        for (auto& v: row_vecs) {
            g.add_edge(result, v);
        }

        return result;
    }

    void row_cat_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;
        assert(g.adj[t->id].size() > 0);

        auto v0 = get_child(t, 0);
        int cols = get_output<la::vector_like<double>>(v0).size();
        int rows = g.adj[t->id].size();

        if (t->output == nullptr) {
            la::matrix<double> m;
            m.resize(rows, cols);
            t->output = std::make_shared<la::matrix<double>>(std::move(m));
        }

        auto& m = get_output<la::matrix_like<double>>(t);

        assert(m.rows() == rows && m.cols() == cols);

        for (int i = 0; i < m.rows(); ++i) {
            auto vi = get_child(t, i);
            auto& v = get_output<la::vector_like<double>>(vi);

            assert(v.size() == cols);

            for (int j = 0; j < m.cols(); ++j) {
                m(i, j) = v(j);
            }
        }
    }

    void row_cat_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& m = autodiff::get_grad<la::matrix_like<double>>(t);

        assert(m.rows() == g.adj[t->id].size());

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto c = get_child(t, i);

            auto& v = autodiff::get_output<la::vector_like<double>>(c);

            assert(v.size() == m.cols());

            if (c->grad_needed && c->grad == nullptr) {
                la::vector<double> g;
                g.resize(v.size());
                c->grad = std::make_shared<la::vector<double>>(g);
            }

            auto& g = autodiff::get_grad<la::vector_like<double>>(c);

            if (c->grad_needed) {
                for (int j = 0; j < m.cols(); ++j) {
                    g(j) += m(i, j);
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

        return result;
    }

    void col_cat_eval(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;
        assert(g.adj[t->id].size() > 0);

        int cols = g.adj[t->id].size();
        auto v0 = get_child(t, 0);
        int rows = get_output<la::vector_like<double>>(v0).size();

        if (t->output == nullptr) {
            la::matrix<double> m;
            m.resize(rows, cols);
            t->output = std::make_shared<la::matrix<double>>(std::move(m));
        }

        auto& m = get_output<la::matrix_like<double>>(t);

        assert(m.rows() == rows && m.cols() == cols);

        for (int j = 0; j < m.cols(); ++j) {
            auto vj = get_child(t, j);
            auto& v = get_output<la::vector_like<double>>(vj);

            assert(v.size() == rows);

            for (int i = 0; i < m.rows(); ++i) {
                m(i, j) = v(i);
            }
        }
    }

    void col_cat_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& m = autodiff::get_grad<la::matrix_like<double>>(t);

        assert(m.cols() == g.adj[t->id].size());

        for (int j = 0; j < g.adj[t->id].size(); ++j) {
            auto c = get_child(t, j);

            auto& v = autodiff::get_output<la::vector_like<double>>(c);

            assert(v.size() == m.rows());

            if (c->grad_needed && c->grad == nullptr) {
                la::vector<double> g;
                g.resize(v.size());
                c->grad = std::make_shared<la::vector<double>>(g);
            }

            auto& g = autodiff::get_grad<la::vector_like<double>>(c);

            if (c->grad_needed) {
                for (int i = 0; i < m.rows(); ++i) {
                    g(i) += m(i, j);
                }
            }
        }
    }

    std::shared_ptr<op_t> row_at(std::shared_ptr<op_t> const& t, int i)
    {
        auto& g = *t->graph;

        std::shared_ptr<op_t> result = g.make_node("row_at");
        result->data = std::make_shared<int>(i);
    
        g.add_edge(result, t);
    
        return result;
    }

    void row_at_eval(std::shared_ptr<op_t> t)
    {
        int i = *std::static_pointer_cast<int>(t->data);
        auto& m = get_output<la::matrix_like<double>>(get_child(t, 0));
        assert(i < m.rows());
        la::weak_vector<double> v { m.data() + m.cols() * i, m.cols() };
        t->output = std::make_shared<la::weak_vector<double>>(v);
    }

    void row_at_grad(std::shared_ptr<op_t> t)
    {
        int i = *std::static_pointer_cast<int>(t->data);
        auto c = get_child(t, 0);

        if (c->grad == nullptr) {
            auto& m = get_output<la::matrix_like<double>>(c);
            la::matrix<double> g;
            g.resize(m.rows(), m.cols());
            c->grad = std::make_shared<la::matrix<double>>(std::move(g));
        }

        auto& v = get_grad<la::vector_like<double>>(t);
        auto& g = get_grad<la::matrix_like<double>>(c);

        for (int d = 0; d < v.size(); ++d) {
            g(i, d) += v(d);
        }
    }

    std::shared_ptr<op_t> corr(std::shared_ptr<op_t> const& t1, std::shared_ptr<op_t> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("corr");

        g.add_edge(result, t1);
        g.add_edge(result, t2);

        return result;
    }

    void corr_eval(std::shared_ptr<op_t> t)
    {
        auto& u = autodiff::get_output<la::tensor_like<double>>(get_child(t, 0));
        auto& v = autodiff::get_output<la::tensor_like<double>>(get_child(t, 1));

        if (t->output == nullptr) {
            la::tensor<double> w;
            w.resize(std::vector<unsigned int> { u.size(0), u.size(1), v.size(3) });
            t->output = std::make_shared<la::tensor<double>>(w);
        }

        la::matrix<double> u_mat;
        u_mat.resize(u.size(0) * u.size(1), v.size(0) * v.size(1) * v.size(2));

        op::corr_linearize(u_mat, u, v.size(0), v.size(1));

        la::weak_matrix<double> v_mat { v.data(), v.size(0) * v.size(1) * v.size(2), v.size(3) };

        auto& result = autodiff::get_output<la::tensor_like<double>>(t);
        la::weak_matrix<double> result_mat { result.data(), result.size(0) * result.size(1), result.size(2) };

        /*
        std::cout << "u" << std::endl;
        for (int i = 0; i < u_mat.rows(); ++i) {
            for (int j = 0; j < u_mat.cols(); ++j) {
                std::cout << u_mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */

        /*
        std::cout << "v" << std::endl;
        for (int i = 0; i < v_mat.rows(); ++i) {
            for (int j = 0; j < v_mat.cols(); ++j) {
                std::cout << v_mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */

        la::mul(result_mat, u_mat, v_mat);
    }

    void corr_grad(std::shared_ptr<op_t> t)
    {
        auto& u = autodiff::get_output<la::tensor_like<double>>(get_child(t, 0));
        auto& v = autodiff::get_output<la::tensor_like<double>>(get_child(t, 1));

        la::matrix<double> u_mat;
        u_mat.resize(u.size(0) * u.size(1), v.size(0) * v.size(1) * v.size(2));

        op::corr_linearize(u_mat, u, v.size(0), v.size(1));

        la::weak_matrix<double> v_mat { v.data(), v.size(0) * v.size(1) * v.size(2), v.size(3) };

        auto u_op = get_child(t, 0);

        if (u_op->grad_needed && u_op->grad == nullptr) {
            la::tensor<double> g;
            g.resize(std::vector<unsigned int> { u.size(0), u.size(1), u.size(2) });
            u_op->grad = std::make_shared<la::tensor<double>>(g);
        }

        auto v_op = get_child(t, 1);

        if (v_op->grad_needed && v_op->grad == nullptr) {
            la::tensor<double> g;
            g.resize(std::vector<unsigned int> { v.size(0), v.size(1), v.size(2), v.size(3) });
            v_op->grad = std::make_shared<la::tensor<double>>(g);
        }

        auto& o = autodiff::get_output<la::tensor_like<double>>(t);

        auto& g_o = autodiff::get_grad<la::tensor_like<double>>(t);
        auto& g_u = autodiff::get_grad<la::tensor_like<double>>(u_op);
        auto& g_v = autodiff::get_grad<la::tensor_like<double>>(v_op);

        la::weak_matrix<double> g_o_mat { g_o.data(), g_o.size(0) * g_o.size(1), g_o.size(2) };

        /*
        std::cout << "grad u" << std::endl;
        for (int i = 0; i < g_o_mat.rows(); ++i) {
            for (int j = 0; j < g_o_mat.cols(); ++j) {
                std::cout << g_o_mat(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        */

        if (u_op->grad_needed) {
            la::matrix<double> g_u_mat;
            g_u_mat.resize(u_mat.rows(), u_mat.cols());
            la::rtmul(g_u_mat, g_o_mat, v_mat);

            /*
            std::cout << "grad u" << std::endl;
            for (int i = 0; i < g_u_mat.rows(); ++i) {
                for (int j = 0; j < g_u_mat.cols(); ++j) {
                    std::cout << g_u_mat(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            */

            op::corr_linearize_grad(g_u, g_u_mat, v.size(0), v.size(1));
        }

        if (v_op->grad_needed) {
            la::weak_matrix<double> g_v_mat { g_v.data(), g_v.size(0) * g_v.size(1) * g_v.size(2), g_v.size(3) };
            la::ltmul(g_v_mat, u_mat, g_o_mat);
        }
    }

    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots)
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
