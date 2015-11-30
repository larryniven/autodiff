#include "autodiff/autodiff.h"
#include <algorithm>
#include "ebt/ebt.h"
#include <cassert>

namespace autodiff {

    op::op()
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

    std::shared_ptr<op> computation_graph::var()
    {
        return make_node("var");
    }

    std::shared_ptr<op> computation_graph::make_node(std::string name)
    {
        std::shared_ptr<op> result { new op };

        result->name = name;
        result->graph = this;
        vertices.push_back(result);
        adj.resize(vertices.size());
        result->id = vertices.size() - 1;

        return result;
    }

    void computation_graph::add_edge(std::shared_ptr<op> const& tail,
        std::shared_ptr<op> const& head)
    {
        adj[tail->id].push_back(head->id);
    }

    std::shared_ptr<op> get_child(std::shared_ptr<op> const& t, int index)
    {
        auto& g = *t->graph;

        return g.vertices[g.adj[t->id][index]];
    }

    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        assert(t1->graph == t2->graph);
        assert(t1->graph != nullptr);

        auto& g = *t1->graph;

        std::shared_ptr<op> result = g.make_node("mult");

        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }
    
    void mult_eval(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        auto& A = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        auto& v = get_output<std::vector<double>>(get_child(t, 1));

        assert(A.size() > 0);
        assert(A.front().size() == v.size());

        if (t->output == nullptr) {
            std::vector<double> z;
            z.resize(A.size());
            t->output = std::make_shared<std::vector<double>>(std::move(z));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        for (int i = 0; i < A.size(); ++i) {
            auto& u = A[i];
            result[i] = 0;

            for (int j = 0; j < v.size(); ++j) {
                result[i] += u[j] * v[j];
            }
        }

    }

    void mult_grad(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        auto& grad = get_grad<std::vector<double>>(t);

        auto& A = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        auto& v = get_output<std::vector<double>>(get_child(t, 1));

        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        if (get_child(t, 1)->grad == nullptr) {
            get_child(t, 1)->grad = std::make_shared<std::vector<double>>(
                std::vector<double>());
        }

        std::vector<std::vector<double>>& A_grad
            = get_grad<std::vector<std::vector<double>>>(get_child(t, 0));
        A_grad.resize(A.size());
        for (int i = 0; i < A.size(); ++i) {
            A_grad.at(i).resize(A.at(i).size());
        }

        std::vector<double>& v_grad
            = get_grad<std::vector<double>>(get_child(t, 1));
        v_grad.resize(v.size());

        #pragma omp parallel for
        for (int j = 0; j < v.size(); ++j) {
            for (int i = 0; i < A.size(); ++i) {
                A_grad[i][j] += grad[i] * v[j];
                v_grad[j] += grad[i] * A[i][j];
            }
        }
    }

    std::shared_ptr<op> logistic(std::shared_ptr<op> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op> result = g.make_node("logistic");
        g.add_edge(result, input);
    
        return result;
    }

    void logistic_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(v.size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        for (int i = 0; i < v.size(); ++i) {
            result[i] = 1.0 / (1.0 + std::exp(-v.at(i)));
        }

    }

    void logistic_grad(std::shared_ptr<op> t)
    {
        auto& grad = get_grad<std::vector<double>>(t);
        auto& output = get_output<std::vector<double>>(t);

        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(get_child(t, 0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += grad[i] * output[i] * (1 - output[i]);
        }
    }

    std::shared_ptr<op> relu(std::shared_ptr<op> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op> result = g.make_node("relu");

        g.add_edge(result, input);
    
        return result;
    }

    void relu_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(v.size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        for (int i = 0; i < v.size(); ++i) {
            result[i] = std::max(0.0, v[i]);
        }
    }

    void relu_grad(std::shared_ptr<op> t)
    {
        auto& output = get_output<std::vector<double>>(t);
        auto& grad = get_grad<std::vector<double>>(t);

        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(get_child(t, 0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += (output[i] > 0 ? grad[i] : 0);
        }
    }

    std::shared_ptr<op> tanh(std::shared_ptr<op> input)
    {
        auto& g = *input->graph;

        std::shared_ptr<op> result = g.make_node("tanh");

        g.add_edge(result, input);
    
        return result;
    }

    void tanh_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(v.size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        for (int i = 0; i < v.size(); ++i) {
            double z1 = std::exp(v[i]);
            double z2 = std::exp(-v[i]);
            result[i] = (z1 - z2) / (z1 + z2);
        }
    }

    void tanh_grad(std::shared_ptr<op> t)
    {
        auto& grad = get_grad<std::vector<double>>(t);
        auto& output = get_output<std::vector<double>>(t);

        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(get_child(t, 0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += grad[i] * (1 - output[i] * output[i]);
        }
    }

    std::shared_ptr<op> add(std::vector<std::shared_ptr<op>> ts)
    {
        auto& g = *ts.front()->graph;

        std::shared_ptr<op> result = g.make_node("add");

        for (auto& t: ts) {
            g.add_edge(result, t);
        }

        return result;
    }

    std::shared_ptr<op> add(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op> result = g.make_node("add");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void add_eval(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        assert(g.adj[t->id].size() > 0);

#ifndef NDEBUG
        for (int i = 1; i < g.adj[t->id].size(); ++i) {
            assert(get_output<std::vector<double>>(get_child(t, i-1)).size()
                == get_output<std::vector<double>>(get_child(t, i)).size());
        }
#endif

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(get_output<std::vector<double>>(get_child(t, 0)).size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto& u = get_output<std::vector<double>>(get_child(t, i));

            for (int j = 0; j < u.size(); ++j) {
                result[j] += u[j];
            }
        }

    }

    void add_grad(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        for (int i = 0; i < g.adj[t->id].size(); ++i) {
            auto c = get_child(t, i);

            if (c->grad == nullptr) {
                std::vector<double> u;
                u.resize(grad.size());
                c->grad = std::make_shared<std::vector<double>>(std::move(u));
            }

            auto& u = get_grad<std::vector<double>>(c);

            for (int i = 0; i < grad.size(); ++i) {
                u[i] += grad[i];
            }
        }
    }
    
    std::shared_ptr<op> softmax(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op> result = g.make_node("softmax");
    
        g.add_edge(result, t);
    
        return result;
    }

    void softmax_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(v.size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        double logZ = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < v.size(); ++j) {
            logZ = ebt::log_add(logZ, v[j]);
        }

        for (int i = 0; i < v.size(); ++i) {
            result[i] = std::exp(v[i] - logZ);
        }

    }

    void softmax_grad(std::shared_ptr<op> t)
    {
        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double> const& output = get_output<std::vector<double>>(t);
        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        std::vector<double>& result = get_grad<std::vector<double>>(get_child(t, 0));
        result.resize(grad.size());

        double Z = 0;
        for (int i = 0; i < grad.size(); ++i) {
            Z += grad[i] * output[i];
        }

        for (int i = 0; i < grad.size(); ++i) {
            result[i] += output[i] * (grad[i] - Z);
        }
    }
    
    std::shared_ptr<op> logsoftmax(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op> result = g.make_node("logsoftmax");
    
        g.add_edge(result, t);
    
        return result;
    }

    void logsoftmax_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<double> result;
            result.resize(v.size());
            t->output = std::make_shared<std::vector<double>>(std::move(result));
        }

        std::vector<double>& result = get_output<std::vector<double>>(t);

        double logZ = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < v.size(); ++j) {
            logZ = ebt::log_add(logZ, v[j]);
        }

        for (int i = 0; i < v.size(); ++i) {
            result[i] = v[i] - logZ;
        }

    }

    void logsoftmax_grad(std::shared_ptr<op> t)
    {
        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double> const& output = get_output<std::vector<double>>(t);
        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        std::vector<double>& result = get_grad<std::vector<double>>(get_child(t, 0));
        result.resize(grad.size());

        double Z = 0;
        for (int i = 0; i < grad.size(); ++i) {
            Z += grad[i];
        }

        for (int i = 0; i < grad.size(); ++i) {
            result[i] += grad[i] - std::exp(output[i]) * Z;
        }
    }
    
    std::shared_ptr<op> transpose(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op> result = g.make_node("transpose");
    
        g.add_edge(result, t);
    
        return result;
    }

    void transpose_eval(std::shared_ptr<op> t)
    {
        auto& A = get_output<std::vector<std::vector<double>>>(get_child(t, 0));

        if (t->output == nullptr) {
            std::vector<std::vector<double>> result;
            result.resize(A.front().size());
            for (auto& v: result) {
                v.resize(A.size());
            }
            t->output = std::make_shared<std::vector<std::vector<double>>>(std::move(result));
        }

        std::vector<std::vector<double>>& result = get_output<std::vector<std::vector<double>>>(t);

        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A[i].size(); ++j) {
                result[j][i] = A[i][j];
            }
        }
    }

    void transpose_grad(std::shared_ptr<op> t)
    {
        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<std::vector<double>> const& grad = get_grad<std::vector<std::vector<double>>>(t);

        std::vector<std::vector<double>>& result = get_grad<std::vector<std::vector<double>>>(get_child(t, 0));
        result.resize(grad.front().size());
        for (auto& v: result) {
            v.resize(grad.size());
        }

        for (int i = 0; i < grad.size(); ++i) {
            for (int j = 0; j < grad[i].size(); ++j) {
                result[j][i] += grad[i][j];
            }
        }
    }
    
    std::shared_ptr<op> conv(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op> result = g.make_node("conv");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void conv_eval(std::shared_ptr<op> t)
    {
        auto& image = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        auto& filter = get_output<std::vector<std::vector<double>>>(get_child(t, 1));

        if (t->output == nullptr) {
            std::vector<std::vector<double>> result;
            result.resize(image.size());
            for (auto& v: result) {
                v.resize(image.front().size());
            }
            t->output = std::make_shared<std::vector<std::vector<double>>>(std::move(result));
        }

        std::vector<std::vector<double>>& result = get_output<std::vector<std::vector<double>>>(t);

        for (int m = 0; m < result.size(); ++m) {
            for (int n = 0; n < result[m].size(); ++n) {
                double sum = 0;
                for (int i = 0; i < filter.size() && 0 <= m-i && m-i < image.size(); ++i) {
                    for (int j = 0; j < filter[i].size() && 0 <= n-j && n-j < image[m-i].size(); ++j) {
                        sum += image[m-i][n-j] * filter[i][j];
                    }
                }
                result[m][n] = sum;
            }
        }
    }

    void conv_grad(std::shared_ptr<op> t)
    {
        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        if (get_child(t, 1)->grad == nullptr) {
            get_child(t, 1)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        std::vector<std::vector<double>> const& grad = get_grad<std::vector<std::vector<double>>>(t);

        std::vector<std::vector<double>> const& image = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        std::vector<std::vector<double>> const& filter = get_output<std::vector<std::vector<double>>>(get_child(t, 1));

        std::vector<std::vector<double>>& image_grad = get_grad<std::vector<std::vector<double>>>(get_child(t, 0));
        image_grad.resize(image.size());
        for (auto& v: image_grad) {
            v.resize(image.front().size());
        }

        for (int i = 0; i < image_grad.size(); ++i) {
            for (int j = 0; j < image_grad[i].size(); ++j) {
                double sum = 0;
                for (int m = i; m < grad.size() && m-i < filter.size(); ++m) {
                    for (int n = j; n < grad[m].size() && n-j < filter[m-i].size(); ++n) {
                        sum += grad[m][n] * filter[m-i][n-j];
                    }
                }
                image_grad[i][j] += sum;
            }
        }

        std::vector<std::vector<double>>& filter_grad = get_grad<std::vector<std::vector<double>>>(get_child(t, 1));
        filter_grad.resize(filter.size());
        for (auto& v: filter_grad) {
            v.resize(filter.front().size());
        }

        for (int i = 0; i < filter_grad.size(); ++i) {
            for (int j = 0; j < filter_grad[i].size(); ++j) {
                double sum = 0;
                for (int m = i; m < grad.size() && m-i < image.size(); ++m) {
                    for (int n = j; n < grad[m].size() && n-j < image[m-i].size(); ++n) {
                        sum += grad[m][n] * image[m-i][n-j];
                    }
                }
                filter_grad[i][j] += sum;
            }
        }
    }

    std::shared_ptr<op> dot(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        auto& g = *t1->graph;

        std::shared_ptr<op> result = g.make_node("dot");
    
        g.add_edge(result, t1);
        g.add_edge(result, t2);
    
        return result;
    }

    void dot_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));
        auto& u = get_output<std::vector<double>>(get_child(t, 1));

        assert(v.size() == u.size());

        double sum = 0;

        for (int i = 0; i < v.size(); ++i) {
            sum += v[i] * u[i];
        }

        t->output = std::make_shared<double>(sum);
    }

    void dot_grad(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(get_child(t, 0));
        auto& u = get_output<std::vector<double>>(get_child(t, 1));

        assert(v.size() == u.size());

        double grad = get_grad<double>(t);

        if (get_child(t, 0)->grad == nullptr) {
            std::vector<double> g;
            g.resize(v.size());
            get_child(t, 0)->grad = std::make_shared<std::vector<double>>(g);
        }

        auto& v_grad = get_grad<std::vector<double>>(get_child(t, 0));

        for (int i = 0; i < v_grad.size(); ++i) {
            v_grad[i] += grad * u[i];
        }

        if (get_child(t, 1)->grad == nullptr) {
            std::vector<double> g;
            g.resize(u.size());
            get_child(t, 1)->grad = std::make_shared<std::vector<double>>(g);
        }

        auto& u_grad = get_grad<std::vector<double>>(get_child(t, 1));

        for (int i = 0; i < u_grad.size(); ++i) {
            u_grad[i] += grad * v[i];
        }
    }

    std::shared_ptr<op> linearize(std::shared_ptr<op> t)
    {
        auto& g = *t->graph;

        std::shared_ptr<op> result = g.make_node("linearize");

        g.add_edge(result, t);

        return result;
    }

    void linearize_eval(std::shared_ptr<op> t)
    {
        if (t->output == nullptr) {
            t->output = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        auto& A = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        auto& result = get_output<std::vector<double>>(t);

        assert(A.size() > 0);

        result.resize(A.size() * A.front().size());

        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A.front().size(); ++j) {
                result[i * A.front().size() + j] = A[i][j];
            }
        }
    }

    void linearize_grad(std::shared_ptr<op> t)
    {
        if (get_child(t, 0)->grad == nullptr) {
            get_child(t, 0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        auto& grad = get_grad<std::vector<double>>(t);
        auto& A = get_output<std::vector<std::vector<double>>>(get_child(t, 0));
        auto& A_grad = get_grad<std::vector<std::vector<double>>>(get_child(t, 1));

        assert(A.size() > 0);
        assert(A.size() * A.front().size() == grad.size());

        A_grad.resize(A.size());
        for (auto& v: A_grad) {
            v.resize(A.front().size());
        }

        for (int i = 0; i < A_grad.size(); ++i) {
            for (int j = 0; j < A_grad[i].size(); ++j) {
                A_grad[i][j] += grad[i * A.front().size() + j];
            }
        }
    }

    std::vector<std::shared_ptr<op>> topo_order(std::shared_ptr<op> const& root)
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

        std::vector<std::shared_ptr<op>> order;

        std::vector<color_t> color;
        color.resize(g.vertices.size(), color_t::white);

        std::vector<std::pair<action_t, std::shared_ptr<op>>> stack;

        stack.push_back(std::make_pair(action_t::color_grey, root));

        while (stack.size() != 0) {
            action_t a;
            std::shared_ptr<op> t;

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

    void eval_vertex(std::shared_ptr<op> const& t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> const& funcs)
    {
        funcs.at(t->name)(t);
    }

    void eval(std::vector<std::shared_ptr<op>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> const& funcs)
    {
        for (int i = topo_order.size() - 1; i >= 0; --i) {
            eval_vertex(topo_order[i], funcs);
        }
    }

    void eval(std::shared_ptr<op> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> const& funcs)
    {
        auto order = topo_order(root);
        eval(order, funcs);
    }

    void grad(std::vector<std::shared_ptr<op>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> const& funcs)
    {
        for (int i = 0; i < topo_order.size(); ++i) {
            eval_vertex(topo_order[i], funcs);
        }
    }

    void grad(std::shared_ptr<op> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> const& funcs)
    {
        auto order = topo_order(root);
        grad(order, funcs);
    }

    void clear_grad(std::vector<std::shared_ptr<op>> const& topo_order)
    {
        for (auto& t: topo_order) {
            t->grad = nullptr;
        }
    }

    void clear_grad(std::shared_ptr<op> const& root)
    {
        auto order = topo_order(root);
        clear_grad(order);
    }

}
