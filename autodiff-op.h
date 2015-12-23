#ifndef AUTODIFF_OP_H
#define AUTODIFF_OP_H

#include "la/la.h"

namespace autodiff {

    namespace op {

        void iouter_prod(la::matrix<double>& result,
            la::vector<double> const& x,
            la::vector<double> const& y);

        void ilmult(la::vector<double>& result,
            la::matrix<double> const& a,
            la::vector<double> const& x);

        la::vector<double> logistic(la::vector<double> const& v);

        void ilogistic_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        la::vector<double> relu(la::vector<double> const& v);

        void irelu_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        la::vector<double> tanh(la::vector<double> const& v);

        void itanh_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        la::vector<double> softmax(la::vector<double> const& v);

        void isoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        la::vector<double> logsoftmax(la::vector<double> const& v);

        void ilogsoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);
    }
}

#endif
