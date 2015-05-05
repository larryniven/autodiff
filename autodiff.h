#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace autodiff {

    struct op {
        std::vector<std::shared_ptr<op>> children;
        op *parent;
    
        std::shared_ptr<void> output;
        std::shared_ptr<void> grad;
    
        std::string name;
    };
    
    template <class T>
    std::shared_ptr<op> var(T&& t)
    {
        std::shared_ptr<op> result { new op };

        result->output = std::make_shared<typename std::remove_reference<T>::type>(std::forward<T>(t));

        result->name = "var";

        return result;
    }

    std::shared_ptr<op> var();
    inline void var_eval(std::shared_ptr<op> t) {};
    inline void var_grad(std::shared_ptr<op> t) {};

    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2);
    void mult_eval(std::shared_ptr<op> t);
    void mult_grad(std::shared_ptr<op> t);

    std::shared_ptr<op> logistic(std::shared_ptr<op> input);
    void logistic_eval(std::shared_ptr<op> t);
    void logistic_grad(std::shared_ptr<op> t);

    std::shared_ptr<op> relu(std::shared_ptr<op> input);
    void relu_eval(std::shared_ptr<op> t);
    void relu_grad(std::shared_ptr<op> t);

    std::shared_ptr<op> add(std::shared_ptr<op> t1, std::shared_ptr<op> t2);
    void add_eval(std::shared_ptr<op> t);
    void add_grad(std::shared_ptr<op> t);

    void eval(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs);

    void grad(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs);

    template <class T>
    T& get_output(std::shared_ptr<op> t)
    {
        return *static_cast<T*>(t->output.get());
    }

    template <class T>
    T& get_grad(std::shared_ptr<op> t)
    {
        return *static_cast<T*>(t->grad.get());
    }

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> eval_funcs {
        { "mult", mult_eval },
        { "logistic", logistic_eval },
        { "relu", relu_eval },
        { "var", var_eval },
        { "add", add_eval }
    };

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> grad_funcs {
        { "mult", mult_grad },
        { "logistic", logistic_grad },
        { "relu", relu_grad },
        { "var", var_grad },
        { "add", add_grad }
    };

}

#endif
