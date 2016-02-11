#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include "cblas.h"
#include <limits>

namespace autodiff {

    namespace op {

        void iouter_prod(la::matrix<double>& result,
            la::vector<double> const& x,
            la::vector<double> const& y)
        {
            result.resize(x.size(), y.size());
            cblas_dger(CblasRowMajor, x.size(), y.size(), 1, x.data(), 1, y.data(), 1,
                result.data(), y.size());
        }

        void ilmul(la::vector<double>& result,
            la::matrix<double> const& a,
            la::vector<double> const& x)
        {
            assert(a.rows() == x.size());

            result.resize(a.cols());
            cblas_dgemv(CblasRowMajor, CblasTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
                x.data(), 1, 1, result.data(), 1);
        }

        void iemul_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& v)
        {
            assert(grad.size() == v.size());

            result.resize(grad.size());
            cblas_dgbmv(CblasRowMajor, CblasNoTrans, grad.size(), grad.size(), 0, 0,
                1.0, grad.data(), 1, v.data(), 1, 1.0, result.data(), 1);
        }

        void logistic(la::vector<double>& u, la::vector<double> const& v)
        {
            assert(u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                u(i) = 1 / (1 + std::exp(-v(i)));
            }
        }

        void ilogistic_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output)
        {
            assert(grad.size() == output.size());

            result.resize(output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += grad(i) * output(i) * (1 - output(i));
            }
        }

        void relu(la::vector<double>& u, la::vector<double> const& v)
        {
            assert(u.size() == v.size());

            for (int i = 0; i < v.size(); ++i) {
                u(i) = std::max(0.0, v(i));
            }
        }

        void irelu_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output)
        {
            assert(grad.size() == output.size());

            result.resize(output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += (output(i) > 0 ? grad(i) : 0);
            }
        }

        void tanh(la::vector<double>& u, la::vector<double> const& v)
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

        void itanh_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output)
        {
            assert(grad.size() == output.size());

            result.resize(output.size());

            for (int i = 0; i < output.size(); ++i) {
                result(i) += grad(i) * (1 - output(i) * output(i));
            }
        }

        void softmax(la::vector<double>& u, la::vector<double> const& v)
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

        void isoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output)
        {
            assert(grad.size() == output.size());

            result.resize(grad.size());

            double mu = la::dot(grad, output);

            for (int i = 0; i < grad.size(); ++i) {
                result(i) += output(i) * (grad(i) - mu);
            }
        }

        void logsoftmax(la::vector<double>& u, la::vector<double> const& v)
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

        void ilogsoftmax_grad(la::vector<double>& result,
            la::vector<double> const& grad,
            la::vector<double> const& output)
        {
            assert(grad.size() == output.size());

            result.resize(grad.size());

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
