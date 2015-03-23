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
    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2);
    std::shared_ptr<op> logistic(std::shared_ptr<op> input);

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
        {
            "mult", [](std::shared_ptr<op> t) {
                auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
                auto& v = get_output<std::vector<double>>(t->children.at(1));

                std::vector<double> result;
                result.resize(A.size());

                for (int i = 0; i < A.size(); ++i) {
                    for (int j = 0; j < v.size(); ++j) {
                        result.at(i) += A.at(i).at(j) * v.at(j);
                    }
                }

                t->output = std::make_shared<std::vector<double>>(std::move(result));
            }
        },
        {
            "logistic", [](std::shared_ptr<op> t) {
                auto& v = get_output<std::vector<double>>(t->children.at(0));

                std::vector<double> result;

                for (int i = 0; i < v.size(); ++i) {
                    result.push_back(1.0 / (1.0 + std::exp(-v.at(i))));
                }

                t->output = std::make_shared<std::vector<double>>(std::move(result));
            }
        },
        {
            "var", [](std::shared_ptr<op> t) {}
        }
    };

    static std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> grad_funcs {
        {
            "mult", [](std::shared_ptr<op> t) {
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

                for (int i = 0; i < A.size(); ++i) {
                    for (int j = 0; j < v.size(); ++j) {
                        A_grad.at(i).at(j) += grad.at(i) * v.at(j);
                        v_grad.at(j) += grad.at(i) * A.at(i).at(j);
                    }
                }

                t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(std::move(A_grad));
                t->children.at(1)->grad = std::make_shared<std::vector<double>>(std::move(v_grad));
            }
        },
        {
            "logistic", [](std::shared_ptr<op> t) {
                auto& output = get_output<std::vector<double>>(t);

                std::vector<double> result;

                for (int i = 0; i < output.size(); ++i) {
                    result.push_back(output.at(i) * (1 - output.at(i)));
                }

                t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::move(result));
            }
        },
        {
            "var", [](std::shared_ptr<op> t) {}
        }
    };
}

#endif
