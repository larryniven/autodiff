#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

namespace autodiff {

    struct computation_graph;

    struct op_t {
        op_t();

        int id;

        std::shared_ptr<void> output;
        std::shared_ptr<void> grad;

        std::shared_ptr<void> memory;
    
        std::string name;

        computation_graph *graph;
    };

    struct computation_graph {
        std::vector<std::shared_ptr<op_t>> vertices; 
        std::vector<std::vector<int>> adj;

        computation_graph();
        computation_graph(computation_graph const& graph);

        computation_graph& operator=(computation_graph const& other);

        template <class T>
        std::shared_ptr<op_t> var(T&& t)
        {
            std::shared_ptr<op_t> result = make_node("var");

            result->output = std::make_shared<typename std::decay<T>::type>(std::forward<T>(t));

            return result;
        }

        std::shared_ptr<op_t> var();

        std::shared_ptr<op_t> make_node(std::string name);

        void add_edge(std::shared_ptr<op_t> const& tail, std::shared_ptr<op_t> const& head);
    };

    std::shared_ptr<op_t> get_child(std::shared_ptr<op_t> const& t, int index);

    inline void var_eval(std::shared_ptr<op_t> t) {};
    inline void var_grad(std::shared_ptr<op_t> t) {};

    std::shared_ptr<op_t> mul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void mul_eval(std::shared_ptr<op_t> t);
    void mul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> emul(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void emul_eval(std::shared_ptr<op_t> t);
    void emul_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> logistic(std::shared_ptr<op_t> input);
    void logistic_eval(std::shared_ptr<op_t> t);
    void logistic_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> relu(std::shared_ptr<op_t> input);
    void relu_eval(std::shared_ptr<op_t> t);
    void relu_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> tanh(std::shared_ptr<op_t> input);
    void tanh_eval(std::shared_ptr<op_t> t);
    void tanh_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> add(std::vector<std::shared_ptr<op_t>> t);
    std::shared_ptr<op_t> add(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void add_eval(std::shared_ptr<op_t> t);
    void add_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> softmax(std::shared_ptr<op_t> t);
    void softmax_eval(std::shared_ptr<op_t> t);
    void softmax_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> logsoftmax(std::shared_ptr<op_t> t);
    void logsoftmax_eval(std::shared_ptr<op_t> t);
    void logsoftmax_grad(std::shared_ptr<op_t> t);

    std::shared_ptr<op_t> dot(std::shared_ptr<op_t> t1, std::shared_ptr<op_t> t2);
    void dot_eval(std::shared_ptr<op_t> t);
    void dot_grad(std::shared_ptr<op_t> t);

    std::vector<std::shared_ptr<op_t>> topo_order(std::vector<std::shared_ptr<op_t>> const& roots);
    std::vector<std::shared_ptr<op_t>> topo_order(std::shared_ptr<op_t> const& root);

    void eval_vertex(std::shared_ptr<op_t> const& t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void eval(std::vector<std::shared_ptr<op_t>> const& topo_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void eval(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void grad(std::vector<std::shared_ptr<op_t>> const& top_to_order,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void grad(std::shared_ptr<op_t> const& root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> const& funcs);

    void clear_grad(std::vector<std::shared_ptr<op_t>> const& topo_order);
    void clear_grad(std::shared_ptr<op_t> const& root);

    template <class T>
    T& get_output(std::shared_ptr<op_t> t)
    {
        return *static_cast<T*>(t->output.get());
    }

    template <class T>
    T& get_grad(std::shared_ptr<op_t> t)
    {
        return *static_cast<T*>(t->grad.get());
    }

    template <class T>
    T& get_memory(std::shared_ptr<op_t> t)
    {
        return *static_cast<T*>(t->memory.get());
    }

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> eval_funcs {
        { "mul", mul_eval },
        { "emul", emul_eval },
        { "logistic", logistic_eval },
        { "relu", relu_eval },
        { "tanh", tanh_eval },
        { "var", var_eval },
        { "add", add_eval },
        { "softmax", softmax_eval },
        { "logsoftmax", logsoftmax_eval },
        { "dot", dot_eval },
    };

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>)>> grad_funcs {
        { "mul", mul_grad },
        { "emul", emul_grad },
        { "logistic", logistic_grad },
        { "relu", relu_grad },
        { "tanh", tanh_grad },
        { "var", var_grad },
        { "add", add_grad },
        { "softmax", softmax_grad },
        { "logsoftmax", logsoftmax_grad },
        { "dot", dot_grad },
    };

}

#endif
