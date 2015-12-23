#include "autodiff/autodiff.h"
#include "la/la.h"
#include "ebt/ebt.h"
#include <algorithm>
#include <cassert>
#include "autodiff/autodiff-op.h"
#include "cblas.h"

namespace autodiff {

    op_t::op_t()
        : output(nullptr), grad(nullptr), memory(nullptr)
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

    std::shared_ptr<op_t> mult(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op_t> result = g.make_node("mult");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void mult_eval(std::shared_ptr<op_t> t)
    {
        auto& A = get_output<la::matrix<double>>(get_child(t, 0));
        auto& v = get_output<la::vector<double>>(get_child(t, 1));

        t->output = std::make_shared<la::vector<double>>(la::mult(A, v));
    }

    void mult_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector<double>>(t);

        auto A_o = get_child(t, 0);
        auto v_o = get_child(t, 1);

        auto& A = get_output<la::matrix<double>>(A_o);
        auto& v = get_output<la::vector<double>>(v_o);

        if (A_o->grad == nullptr) {
            A_o->grad = std::make_shared<la::matrix<double>>(la::matrix<double>());
        }

        if (v_o->grad == nullptr) {
            v_o->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& A_grad = get_grad<la::matrix<double>>(A_o);
        auto& v_grad = get_grad<la::vector<double>>(v_o);

        op::iouter_prod(A_grad, grad, v);
        op::ilmult(v_grad, A, grad);
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));

        t->output = std::make_shared<la::vector<double>>(op::logistic(v));
    }

    void logistic_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector<double>>(t);
        auto& output = get_output<la::vector<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            ch->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_grad<la::vector<double>>(ch);
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));

        t->output = std::make_shared<la::vector<double>>(op::relu(v));
    }

    void relu_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector<double>>(t);
        auto& grad = get_grad<la::vector<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            ch->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_grad<la::vector<double>>(ch);
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));

        t->output = std::make_shared<la::vector<double>>(op::tanh(v));
    }

    void tanh_grad(std::shared_ptr<op_t> t)
    {
        auto& grad = get_grad<la::vector<double>>(t);
        auto& output = get_output<la::vector<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            ch->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_grad<la::vector<double>>(ch);
        op::itanh_grad(result, grad, output);
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
            assert(get_output<la::vector<double>>(get_child(t, i-1)).size()
                == get_output<la::vector<double>>(get_child(t, i)).size());
        }
#endif

        if (t->output == nullptr) {
            t->output = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_output<la::vector<double>>(t);
        result.resize(get_output<la::vector<double>>(get_child(t, 0)).size());

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto& u = get_output<la::vector<double>>(get_child(t, i));

            la::iadd(result, u);
        }
    }

    void add_grad(std::shared_ptr<op_t> t)
    {
        auto& g = *t->graph;

        auto& grad = get_grad<la::vector<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto c = get_child(t, i);

            if (c->grad == nullptr) {
                c->grad = std::make_shared<la::vector<double>>(la::vector<double>());
            }

            auto& u = get_grad<la::vector<double>>(c);
            u.resize(grad.size());

            la::iadd(u, grad);
        }
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));

        t->output = std::make_shared<la::vector<double>>(op::softmax(v));
    }

    void softmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector<double>>(t);
        auto& grad = get_grad<la::vector<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            ch->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_grad<la::vector<double>>(ch);
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));

        t->output = std::make_shared<la::vector<double>>(op::logsoftmax(v));
    }

    void logsoftmax_grad(std::shared_ptr<op_t> t)
    {
        auto& output = get_output<la::vector<double>>(t);
        auto& grad = get_grad<la::vector<double>>(t);

        auto ch = get_child(t, 0);
        if (ch->grad == nullptr) {
            ch->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& result = get_grad<la::vector<double>>(ch);
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
        auto& v = get_output<la::vector<double>>(get_child(t, 0));
        auto& u = get_output<la::vector<double>>(get_child(t, 1));

        t->output = std::make_shared<double>(la::dot(v, u));
    }

    void dot_grad(std::shared_ptr<op_t> t)
    {
        auto c0 = get_child(t, 0);
        auto c1 = get_child(t, 1);

        auto& v = get_output<la::vector<double>>(c0);
        auto& u = get_output<la::vector<double>>(c1);

        assert(v.size() == u.size());

        double grad = get_grad<double>(t);

        if (c0->grad == nullptr) {
            c0->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& v_grad = get_grad<la::vector<double>>(c0);
        v_grad.resize(u.size());

        cblas_daxpy(v_grad.size(), grad, u.data(), 1, v_grad.data(), 1);

        if (c1->grad == nullptr) {
            c1->grad = std::make_shared<la::vector<double>>(la::vector<double>());
        }

        auto& u_grad = get_grad<la::vector<double>>(c1);
        u_grad.resize(v.size());

        cblas_daxpy(u_grad.size(), grad, v.data(), 1, u_grad.data(), 1);
    }

    std::vector<std::shared_ptr<op_t>> topo_order(std::shared_ptr<op_t> const& root)
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

        auto& g = *root->graph;

        std::vector<std::shared_ptr<op_t>> order;

        std::vector<color_t> color;
        color.resize(g.vertices.size(), color_t::white);

        std::vector<std::pair<action_t, std::shared_ptr<op_t>>> stack;

        stack.push_back(std::make_pair(action_t::color_grey, root));

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

    void eval_vertex(std::shared_ptr<op_t> const& t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs)
    {
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
            eval_vertex(topo_order[i], funcs);
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
