#ifndef AUTODIFF_GPU_H
#define AUTODIFF_GPU_H

#include "autodiff/autodiff.h"
#include "la/la-gpu.h"

namespace autodiff {

    namespace gpu {

        template <class T>
        struct memory_pool {

            la::gpu::vector<T> pool;
            unsigned long alloced;

            memory_pool()
                : alloced(0)
            {}

            memory_pool(unsigned long size)
                : alloced(0)
            {
                pool.resize(size);
            }

            double* alloc(unsigned long size)
            {
                if (alloced + size > pool.size()) {
                    std::cerr << "not enough memory in the pool. allocated: " <<  alloced
                        << " requested: " << size << std::endl;
                    exit(1);
                }

                double* result = pool.data() + alloced;
                alloced += size;

                return result;
            }

            void resize(unsigned long size)
            {
                pool.resize(size);
            }

            void reset()
            {
                alloced = 0;
            }

        };

        void var_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void mul_eval(std::shared_ptr<op_t> t);
        void mul_grad(std::shared_ptr<op_t> t);
        void mul_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void emul_eval(std::shared_ptr<op_t> t);
        void emul_grad(std::shared_ptr<op_t> t);
        void emul_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void logistic_eval(std::shared_ptr<op_t> t);
        void logistic_grad(std::shared_ptr<op_t> t);
        void logistic_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void relu_eval(std::shared_ptr<op_t> t);
        void relu_grad(std::shared_ptr<op_t> t);
        void relu_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void tanh_eval(std::shared_ptr<op_t> t);
        void tanh_grad(std::shared_ptr<op_t> t);
        void tanh_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void add_eval(std::shared_ptr<op_t> t);
        void add_grad(std::shared_ptr<op_t> t);
        void add_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void softmax_eval(std::shared_ptr<op_t> t);
        void softmax_grad(std::shared_ptr<op_t> t);
        void softmax_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void logsoftmax_eval(std::shared_ptr<op_t> t);
        void logsoftmax_grad(std::shared_ptr<op_t> t);
        void logsoftmax_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

        void dot_eval(std::shared_ptr<op_t> t);
        void dot_grad(std::shared_ptr<op_t> t);
        void dot_alloc(std::shared_ptr<op_t> t, memory_pool<double>& mem);

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

        static std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
                memory_pool<double>&)>> alloc_funcs {
            { "mul", mul_alloc },
            { "emul", emul_alloc },
            { "logistic", logistic_alloc },
            { "relu", relu_alloc },
            { "tanh", tanh_alloc },
            { "var", var_alloc },
            { "add", add_alloc },
            { "softmax", softmax_alloc },
            { "logsoftmax", logsoftmax_alloc },
            { "dot", dot_alloc },
        };

        void alloc_vertex(std::shared_ptr<autodiff::op_t> const& t,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs);

        void alloc(std::vector<std::shared_ptr<op_t>> const& topo_order,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs);

        void alloc(std::shared_ptr<op_t> const& root,
            memory_pool<double>& mem,
            std::unordered_map<std::string, std::function<void(std::shared_ptr<op_t>,
            memory_pool<double>&)>> const& funcs);

    }
}

#endif
