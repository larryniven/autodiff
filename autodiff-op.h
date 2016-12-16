#ifndef AUTODIFF_OP_H
#define AUTODIFF_OP_H

#include "la/la.h"

namespace autodiff {

    namespace op {

        void logistic(la::vector_like<double>& u, la::vector_like<double> const& v);

        void ilogistic_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output);

        void relu(la::vector_like<double>& u, la::vector_like<double> const& v);

        void irelu_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output);

        void tanh(la::vector_like<double>& u, la::vector_like<double> const& v);

        void itanh_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output);

        void softmax(la::vector_like<double>& u, la::vector_like<double> const& v);

        void isoftmax_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output);

        void logsoftmax(la::vector_like<double>& u, la::vector_like<double> const& v);

        void ilogsoftmax_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output);

        void corr_linearize(la::matrix_like<double>& result,
            la::tensor_like<double> const& u,
            int f1, int f2);

        void corr_linearize_grad(la::tensor_like<double>& result,
            la::matrix_like<double> const& u,
            int f1, int f2);
    }
}

#endif
