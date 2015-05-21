#include "autodiff/autodiff.h"
#include <algorithm>
#include "ebt/ebt.h"

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

        for (int i = 0; i < A.size(); ++i) {
            auto& u = A[i];

            double* u_data = u.data();
            double* v_data = v.data();
            double* result_data = result.data();
            int size = v.size();

            for (int j = 0; j < size; ++j) {
                result_data[i] += u_data[j] * v_data[j];
            }
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void mult_grad(std::shared_ptr<op> t)
    {
        auto& grad = get_grad<std::vector<double>>(t);

        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& v = get_output<std::vector<double>>(t->children.at(1));

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        if (t->children.at(1)->grad == nullptr) {
            t->children.at(1)->grad = std::make_shared<std::vector<double>>(
                std::vector<double>());
        }

        std::vector<std::vector<double>>& A_grad
            = get_grad<std::vector<std::vector<double>>>(t->children.at(0));
        A_grad.resize(A.size());
        for (int i = 0; i < A.size(); ++i) {
            A_grad.at(i).resize(A.at(i).size());
        }

        std::vector<double>& v_grad
            = get_grad<std::vector<double>>(t->children.at(1));
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

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += output[i] * (1 - output[i]);
        }
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
            result.push_back(std::max(0.0, v[i]));
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void relu_grad(std::shared_ptr<op> t)
    {
        auto& output = get_output<std::vector<double>>(t);
        auto& grad = get_grad<std::vector<double>>(t);

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += (output[i] > 0 ? grad[i] : 0);
        }
    }

    std::shared_ptr<op> add(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        std::shared_ptr<op> result { new op };
    
        t1->parent = result.get();
        t2->parent = result.get();
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "add";
    
        return result;
    }

    void add_eval(std::shared_ptr<op> t)
    {
        auto& u = get_output<std::vector<double>>(t->children.at(0));
        auto& v = get_output<std::vector<double>>(t->children.at(1));

        std::vector<double> result;
        result.resize(u.size());

        for (int j = 0; j < u.size(); ++j) {
            result[j] = u[j] + v[j];
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void add_grad(std::shared_ptr<op> t)
    {
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        if (t->children.at(1)->grad == nullptr) {
            t->children.at(1)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        std::vector<double>& left = get_grad<std::vector<double>>(t->children.at(0));
        left.resize(grad.size());
        std::vector<double>& right = get_grad<std::vector<double>>(t->children.at(1));
        right.resize(grad.size());

        for (int i = 0; i < grad.size(); ++i) {
            left[i] += grad[i];
            right[i] += grad[i];
        }
    }
    
    std::shared_ptr<op> logsoftmax(std::shared_ptr<op> t)
    {
        std::shared_ptr<op> result { new op };
    
        t->parent = result.get();
    
        result->children.emplace_back(t);
    
        result->name = "logsoftmax";
    
        return result;
    }

    void logsoftmax_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

        std::vector<double> result;
        result.resize(v.size());

        double logZ = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < v.size(); ++j) {
            logZ = ebt::log_add(logZ, v[j]);
        }

        for (int i = 0; i < v.size(); ++i) {
            result[i] = v[i] - logZ;
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void logsoftmax_grad(std::shared_ptr<op> t)
    {
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double> const& output = get_output<std::vector<double>>(t);
        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
        result.resize(grad.size());

        double Z = 0;
        for (int i = 0; i < grad.size(); ++i) {
            Z += grad[i];
        }

        for (int i = 0; i < grad.size(); ++i) {
            result[i] = result[i] - std::exp(output[i]) * Z;
        }
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
