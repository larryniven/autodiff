#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <cblas.h>

namespace autodiff {

    namespace op {

        void iouter_prod(la::matrix_like<double>& result,
            la::vector_like<double> const& x,
            la::vector_like<double> const& y)
        {
            cblas_dger(CblasRowMajor, x.size(), y.size(), 1, x.data(), 1, y.data(), 1,
                result.data(), y.size());
        }

        void ilmul(la::vector_like<double>& result,
            la::matrix_like<double> const& a,
            la::vector_like<double> const& x)
        {
            assert(a.rows() == x.size());

            cblas_dgemv(CblasRowMajor, CblasTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
                x.data(), 1, 1, result.data(), 1);
        }

        void logistic(la::vector_like<double>& u, la::vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                u(i) = 1 / (1 + std::exp(-v(i)));
            }
        }

        void ilogistic_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output)
        {
            assert(grad.size() == output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += grad(i) * output(i) * (1 - output(i));
            }
        }

        void relu(la::vector_like<double>& u, la::vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                u(i) = std::max(0.0, v(i));
            }
        }

        void irelu_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output)
        {
            assert(grad.size() == output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += (output(i) > 0 ? grad(i) : 0);
            }
        }

        void tanh(la::vector_like<double>& u, la::vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                if (v(i) > 0) {
                    double z = std::exp(-2 * v(i));
                    u(i) = (1 - z) / (1 + z);
                } else {
                    double z = std::exp(2 * v(i));
                    u(i) = (z - 1) / (z + 1);
                }
            }
        }

        void itanh_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output)
        {
            assert(grad.size() == output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += grad(i) * (1 - output(i) * output(i));
            }
        }

        void softmax(la::vector_like<double>& u, la::vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            double logZ = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < v.size(); ++j) {
                logZ = ebt::log_add(logZ, v(j));
            }

            for (int i = 0; i < v.size(); ++i) {
                u(i) = std::exp(v(i) - logZ);
            }
        }

        void isoftmax_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output)
        {
            assert(grad.size() == output.size());

            double mu = la::dot(grad, output);

            for (int i = 0; i < grad.size(); ++i) {
                result(i) += output(i) * (grad(i) - mu);
            }
        }

        void logsoftmax(la::vector_like<double>& u, la::vector_like<double> const& v)
        {
            assert(u.size() == v.size());

            double logZ = -std::numeric_limits<double>::infinity();

            for (int j = 0; j < v.size(); ++j) {
                logZ = ebt::log_add(logZ, v(j));
            }

            for (int i = 0; i < v.size(); ++i) {
                u(i) = v(i) - logZ;
            }
        }

        void ilogsoftmax_grad(la::vector_like<double>& result,
            la::vector_like<double> const& grad,
            la::vector_like<double> const& output)
        {
            assert(grad.size() == output.size());

            double mu = 0;
            for (int i = 0; i < grad.size(); ++i) {
                mu += grad(i);
            }

            for (int i = 0; i < grad.size(); ++i) {
                result(i) += grad(i) - std::exp(output(i)) * mu;
            }
        }
    }
}
