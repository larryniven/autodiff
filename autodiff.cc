#include "autodiff/autodiff.h"
#include <algorithm>

namespace autodiff {

    std::shared_ptr<op> var()
    {
        std::shared_ptr<op> result { new op };

        result->name = "var";

        return result;
    }

    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        std::shared_ptr<op> result { new op };
    
        t1->parent = result.get();
        t2->parent = result.get();
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "mult";
    
        return result;
    }
    
    void mult_eval(std::shared_ptr<op> t)
    {
        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& v = get_output<std::vector<double>>(t->children.at(1));

        std::vector<double> result;
        result.resize(A.size());

        #pragma omp parallel for
        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < v.size(); ++j) {
                result.at(i) += A.at(i).at(j) * v.at(j);
            }
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void mult_grad(std::shared_ptr<op> t)
    {
        auto& grad = get_grad<std::vector<double>>(t);

        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& v = get_output<std::vector<double>>(t->children.at(1));

        std::vector<std::vector<double>> A_grad;
        A_grad.resize(A.size());
        for (int i = 0; i < A.size(); ++i) {
            A_grad.at(i).resize(A.at(i).size());
        }

        std::vector<double> v_grad;
        v_grad.resize(v.size());

        #pragma omp parallel for
        for (int j = 0; j < v.size(); ++j) {
            for (int i = 0; i < A.size(); ++i) {
                A_grad.at(i).at(j) += grad.at(i) * v.at(j);
                v_grad.at(j) += grad.at(i) * A.at(i).at(j);
            }
        }

        t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(std::move(A_grad));
        t->children.at(1)->grad = std::make_shared<std::vector<double>>(std::move(v_grad));
    }

    std::shared_ptr<op> logistic(std::shared_ptr<op> input)
    {
        std::shared_ptr<op> result { new op };

        input->parent = result.get();

        result->children.emplace_back(input);
    
        result->name = "logistic";
    
        return result;
    }

    void logistic_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

        std::vector<double> result;

        for (int i = 0; i < v.size(); ++i) {
            result.push_back(1.0 / (1.0 + std::exp(-v.at(i))));
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void logistic_grad(std::shared_ptr<op> t)
    {
        auto& output = get_output<std::vector<double>>(t);

        std::vector<double> result;

        for (int i = 0; i < output.size(); ++i) {
            result.push_back(output.at(i) * (1 - output.at(i)));
        }

        t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::move(result));
    }

    std::shared_ptr<op> relu(std::shared_ptr<op> input)
    {
        std::shared_ptr<op> result { new op };

        input->parent = result.get();

        result->children.emplace_back(input);
    
        result->name = "relu";
    
        return result;
    }

    void relu_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

        std::vector<double> result;

        for (int i = 0; i < v.size(); ++i) {
            result.push_back(std::max(0.0, v.at(i)));
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void relu_grad(std::shared_ptr<op> t)
    {
        auto& output = get_output<std::vector<double>>(t);
        auto& grad = get_grad<std::vector<double>>(t);

        std::vector<double> result;

        for (int i = 0; i < output.size(); ++i) {
            result.push_back(output.at(i) > 0 ? grad.at(i) : 0);
        }

        t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::move(result));
    }

    void eval(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        std::vector<std::shared_ptr<op>> stack { root };
        std::vector<std::shared_ptr<op>> path;

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            while (path.size() > 0 && path.back().get() != t->parent) {
                funcs.at(path.back()->name)(path.back());
                path.pop_back();
            }
            path.push_back(t);

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }

        for (int i = path.size() - 1; i >= 0; --i) {
            funcs.at(path.at(i)->name)(path.at(i));
        }
    }

    void grad(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        std::vector<std::shared_ptr<op>> stack { root };

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            funcs.at(t->name)(t);

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }
    }

}
