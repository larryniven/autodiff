#ifndef AUTODIFF_OP_H
#define AUTODIFF_OP_H

#include "la/la.h"

namespace autodiff {

    namespace op {

        void iouter_prod(la::matrix<double>& result,
            la::vector<double> const& x,
            la::vector<double> const& y);

        void ilmul(la::vector<double>& result,
            la::matrix<double> const& a,
            la::vector<double> const& x);

        void iemul_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& v);

        void logistic(la::vector<double>& u, la::vector<double> const& v);

        void ilogistic_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        void relu(la::vector<double>& u, la::vector<double> const& v);

        void irelu_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        void tanh(la::vector<double>& u, la::vector<double> const& v);

        void itanh_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        void softmax(la::vector<double>& u, la::vector<double> const& v);

        void isoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);

        void logsoftmax(la::vector<double>& u, la::vector<double> const& v);

        void ilogsoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output);
    }
}

#endif
