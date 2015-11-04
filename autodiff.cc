#include "autodiff/autodiff.h"
#include <algorithm>
#include "ebt/ebt.h"
#include <cassert>

namespace autodiff {

    op::op()
        : output(nullptr), grad(nullptr), memory(nullptr)
    {}

    std::shared_ptr<op> var()
    {
        std::shared_ptr<op> result { new op };

        result->name = "var";

        return result;
    }

    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "mult";
    
        return result;
    }
    
    void mult_eval(std::shared_ptr<op> t)
    {
        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& v = get_output<std::vector<double>>(t->children.at(1));

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

        result->children.emplace_back(input);
    
        result->name = "logistic";
    
        return result;
    }

    void logistic_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

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

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += grad[i] * output[i] * (1 - output[i]);
        }
    }

    std::shared_ptr<op> logistic2d(std::shared_ptr<op> input)
    {
        std::shared_ptr<op> result { new op };

        result->children.emplace_back(input);
    
        result->name = "logistic2d";
    
        return result;
    }

    void logistic2d_eval(std::shared_ptr<op> t)
    {
        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));

        assert(A.size() > 0);

        if (t->output == nullptr) {
            std::vector<std::vector<double>> result;
            result.resize(A.size());
            for (auto& v: result) {
                v.resize(A.front().size());
            }
            t->output = std::make_shared<std::vector<std::vector<double>>>(std::move(result));
        }

        std::vector<std::vector<double>>& result = get_output<std::vector<std::vector<double>>>(t);

        for (int i = 0; i < result.size(); ++i) {
            for (int j = 0; j < result[i].size(); ++j) {
                result[i][j] = 1.0 / (1.0 + std::exp(-A[i][j]));
            }
        }

    }

    void logistic2d_grad(std::shared_ptr<op> t)
    {
        auto& output = get_output<std::vector<std::vector<double>>>(t);

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        std::vector<std::vector<double>>& result = get_grad<std::vector<std::vector<double>>>(t->children.at(0));
        result.resize(output.size());
        for (auto& v: result) {
            v.resize(output.front().size());
        }

        for (int i = 0; i < result.size(); ++i) {
            for (int j = 0; j < result[i].size(); ++j) {
                result[i][j] += output[i][j] * (1 - output[i][j]);
            }
        }
    }

    std::shared_ptr<op> relu(std::shared_ptr<op> input)
    {
        std::shared_ptr<op> result { new op };

        result->children.emplace_back(input);
    
        result->name = "relu";
    
        return result;
    }

    void relu_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

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

        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
        result.resize(output.size());

        for (int i = 0; i < output.size(); ++i) {
            result[i] += (output[i] > 0 ? grad[i] : 0);
        }
    }

    std::shared_ptr<op> add(std::vector<std::shared_ptr<op>> ts)
    {
        std::shared_ptr<op> result { new op };
        result->children = std::move(ts);

        result->name = "add";

        return result;
    }

    std::shared_ptr<op> add(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "add";
    
        return result;
    }

    void add_eval(std::shared_ptr<op> t)
    {
        assert(t->children.size() > 0);

#ifndef NDEBUG
        for (int i = 1; i < t->children.size(); ++i) {
            assert(get_output<std::vector<double>>(t->children.at(i-1)).size()
                == get_output<std::vector<double>>(t->children.at(i)).size());
        }
#endif

        std::vector<double> result;
        result.resize(get_output<std::vector<double>>(t->children.front()).size());

        for (auto& c: t->children) {
            auto& u = get_output<std::vector<double>>(c);

            for (int j = 0; j < u.size(); ++j) {
                result[j] += u[j];
            }
        }

        t->output = std::make_shared<std::vector<double>>(std::move(result));
    }

    void add_grad(std::shared_ptr<op> t)
    {
        for (auto& c: t->children) {
            if (c->grad == nullptr) {
                c->grad = std::make_shared<std::vector<double>>(std::vector<double>());
            }
        }

        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        for (auto& c: t->children) {
            auto& u = get_grad<std::vector<double>>(c);
            u.resize(grad.size());
        }

        for (auto& c: t->children) {
            auto& u = get_grad<std::vector<double>>(c);

            for (int i = 0; i < grad.size(); ++i) {
                u[i] += grad[i];
            }
        }
    }
    
    std::shared_ptr<op> softmax(std::shared_ptr<op> t)
    {
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t);
    
        result->name = "softmax";
    
        return result;
    }

    void softmax_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

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
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<double> const& output = get_output<std::vector<double>>(t);
        std::vector<double> const& grad = get_grad<std::vector<double>>(t);

        std::vector<double>& result = get_grad<std::vector<double>>(t->children.at(0));
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
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t);
    
        result->name = "logsoftmax";
    
        return result;
    }

    void logsoftmax_eval(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children.at(0));

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
            result[i] += grad[i] - std::exp(output[i]) * Z;
        }
    }
    
    std::shared_ptr<op> transpose(std::shared_ptr<op> t)
    {
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t);
    
        result->name = "transpose";
    
        return result;
    }

    void transpose_eval(std::shared_ptr<op> t)
    {
        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));

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
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        std::vector<std::vector<double>> const& grad = get_grad<std::vector<std::vector<double>>>(t);

        std::vector<std::vector<double>>& result = get_grad<std::vector<std::vector<double>>>(t->children.at(0));
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
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "conv";
    
        return result;
    }

    void conv_eval(std::shared_ptr<op> t)
    {
        auto& image = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& filter = get_output<std::vector<std::vector<double>>>(t->children.at(1));

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
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        if (t->children.at(1)->grad == nullptr) {
            t->children.at(1)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        std::vector<std::vector<double>> const& grad = get_grad<std::vector<std::vector<double>>>(t);

        std::vector<std::vector<double>> const& image = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        std::vector<std::vector<double>> const& filter = get_output<std::vector<std::vector<double>>>(t->children.at(1));

        std::vector<std::vector<double>>& image_grad = get_grad<std::vector<std::vector<double>>>(t->children.at(0));
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

        std::vector<std::vector<double>>& filter_grad = get_grad<std::vector<std::vector<double>>>(t->children.at(1));
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
        std::shared_ptr<op> result { new op };
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "dot";
    
        return result;
    }

    void dot_eval(std::shared_ptr<op> t)
    {
        if (t->output == nullptr) {
            t->output = std::make_shared<double>(0.0);
        }

        auto& v = get_output<std::vector<double>>(t->children[0]);
        auto& u = get_output<std::vector<double>>(t->children[1]);

        assert(v.size() == u.size());

        double sum = 0;

        for (int i = 0; i < v.size(); ++i) {
            sum += v[i] * u[i];
        }

        t->output = std::make_shared<double>(sum);
    }

    void dot_grad(std::shared_ptr<op> t)
    {
        auto& v = get_output<std::vector<double>>(t->children[0]);
        auto& u = get_output<std::vector<double>>(t->children[1]);

        assert(v.size() == u.size());

        double grad = get_grad<double>(t);

        if (t->children[0]->grad == nullptr) {
            std::vector<double> g;
            g.resize(v.size());
            t->children[0]->grad = std::make_shared<std::vector<double>>(g);
        }

        auto& v_grad = get_grad<std::vector<double>>(t->children[0]);

        for (int i = 0; i < v_grad.size(); ++i) {
            v_grad[i] += grad * u[i];
        }

        if (t->children[1]->grad == nullptr) {
            std::vector<double> g;
            g.resize(u.size());
            t->children[1]->grad = std::make_shared<std::vector<double>>(g);
        }

        auto& u_grad = get_grad<std::vector<double>>(t->children[1]);

        for (int i = 0; i < u_grad.size(); ++i) {
            u_grad[i] += grad * v[i];
        }
    }

    std::shared_ptr<op> linearize(std::shared_ptr<op> t)
    {
        std::shared_ptr<op> result { new op };

        result->children.emplace_back(t);

        result->name = "linearize";

        return result;
    }

    void linearize_eval(std::shared_ptr<op> t)
    {
        if (t->output == nullptr) {
            t->output = std::make_shared<std::vector<double>>(std::vector<double>());
        }

        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
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
        if (t->children.at(0)->grad == nullptr) {
            t->children.at(0)->grad = std::make_shared<std::vector<std::vector<double>>>(
                std::vector<std::vector<double>>());
        }

        auto& grad = get_grad<std::vector<double>>(t);
        auto& A = get_output<std::vector<std::vector<double>>>(t->children.at(0));
        auto& A_grad = get_grad<std::vector<std::vector<double>>>(t->children.at(0));

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

    void clear_output(std::shared_ptr<op> root)
    {
        std::vector<std::shared_ptr<op>> stack { root };
        std::vector<std::shared_ptr<op>> path;

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            t->output = nullptr;

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }
    }

    void clear_grad(std::shared_ptr<op> root)
    {
        std::vector<std::shared_ptr<op>> stack { root };
        std::vector<std::shared_ptr<op>> path;

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            t->grad = nullptr;

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }
    }

    void eval_vertex(std::shared_ptr<op> t,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        funcs.at(t->name)(t);
    }

    void eval(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        std::vector<std::shared_ptr<op>> stack { root };
        std::vector<std::shared_ptr<op>> path;

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();

            // std::cout << "t: " << t->name << std::endl;

            // std::cout << "path: ";
            // for (auto& c: path) {
            //     std::cout << c->name << " (" << c.get() << ") ";
            // }
            // std::cout << std::endl;

            // std::cout << "stack: " << std::endl;
            // for (auto& c: stack) {
            //     std::cout << "  " << c->name << std::endl;
            // }

            stack.pop_back();

            auto is_parent = [](std::shared_ptr<autodiff::op> p, std::shared_ptr<autodiff::op> t) {
                for (auto& c: p->children) {
                    if (c.get() == t.get()) {
                        return true;
                    }
                }
                return false;
            };

            while (path.size() > 0 && !is_parent(path.back(), t)) {
                // std::cout << path.back()->name << std::endl;
                funcs.at(path.back()->name)(path.back());
                path.pop_back();
            }
            path.push_back(t);

            for (auto c: t->children) {
                // std::cout << "child of " << t->name << ": " << c->name << std::endl;
                stack.push_back(c);
            }
        }

        for (int i = path.size() - 1; i >= 0; --i) {
            // std::cout << path.at(i)->name << std::endl;
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
