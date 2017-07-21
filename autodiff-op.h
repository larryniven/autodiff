#ifndef AUTODIFF_OP_H
#define AUTODIFF_OP_H

#include "la/la-cpu.h"

namespace autodiff {

    namespace op {

        void logistic(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v);

        void ilogistic_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output);

        void relu(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v);

        void irelu_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output);

        void tanh(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v);

        void itanh_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output);

        void softmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v);

        void isoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output);

        void logsoftmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v);

        void ilogsoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output);

        void corr_linearize_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& u,
            int f1, int f2, int p1, int p2, int d1, int d2);

        void pool_max(la::cpu::tensor_like<double>& indices,
            la::cpu::tensor_like<double>& output,
            la::cpu::tensor_like<double> const& input,
            int dim1, int dim2, int stride1, int stride2);

        void pool_max_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& indices,
            int dim1, int dim2, int stride1, int stride2);

    }
}

#endif
