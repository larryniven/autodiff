#ifndef AUTODIFF_OP_H
#define AUTODIFF_OP_H

#include "la/la.h"

namespace autodiff {

    namespace op {

        void iouter_prod(la::matrix_like<double>& result,
            la::vector_like<double> const& x,
            la::vector_like<double> const& y);

        void ilmul(la::vector_like<double>& result,
            la::matrix_like<double> const& a,
            la::vector_like<double> const& x);

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
    }
}

#endif
