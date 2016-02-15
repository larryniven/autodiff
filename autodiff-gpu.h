#ifndef AUTODIFF_GPU_H
#define AUTODIFF_GPU_H

#include "autodiff/autodiff.h"

namespace autodiff {

    namespace gpu {

        void mul_eval(std::shared_ptr<op_t> t);
        void mul_grad(std::shared_ptr<op_t> t);

        void emul_eval(std::shared_ptr<op_t> t);
        void emul_grad(std::shared_ptr<op_t> t);

        void logistic_eval(std::shared_ptr<op_t> t);
        void logistic_grad(std::shared_ptr<op_t> t);

        void relu_eval(std::shared_ptr<op_t> t);
        void relu_grad(std::shared_ptr<op_t> t);

        void tanh_eval(std::shared_ptr<op_t> t);
        void tanh_grad(std::shared_ptr<op_t> t);

        void add_eval(std::shared_ptr<op_t> t);
        void add_grad(std::shared_ptr<op_t> t);

        void softmax_eval(std::shared_ptr<op_t> t);
        void softmax_grad(std::shared_ptr<op_t> t);

        void logsoftmax_eval(std::shared_ptr<op_t> t);
        void logsoftmax_grad(std::shared_ptr<op_t> t);

        void dot_eval(std::shared_ptr<op_t> t);
        void dot_grad(std::shared_ptr<op_t> t);

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
}

#endif
