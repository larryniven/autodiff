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

        void ilmult(la::vector<double>& result,
            la::matrix<double> const& a,
            la::vector<double> const& x)
        {
            result.resize(a.cols());
            cblas_dgemv(CblasRowMajor, CblasTrans, a.rows(), a.cols(), 1, a.data(), a.cols(),
                x.data(), 1, 1, result.data(), 1);
        }

        la::vector<double> logistic(la::vector<double> const& v)
        {
            la::vector<double> result;
            result.resize(v.size());

            for (int i = 0; i < v.size(); ++i) {
                result(i) = 1 / (1 + std::exp(-v(i)));
            }

            return result;
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

        la::vector<double> relu(la::vector<double> const& v)
        {
            la::vector<double> result;
            result.resize(v.size());

            for (int i = 0; i < v.size(); ++i) {
                result(i) = std::max(0.0, v(i));
            }

            return result;
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

        la::vector<double> tanh(la::vector<double> const& v)
        {
            la::vector<double> result;
            result.resize(v.size());

            for (int i = 0; i < v.size(); ++i) {
                double z1 = std::exp(v(i));
                double z2 = std::exp(-v(i));
                result(i) = (z1 - z2) / (z1 + z2);
            }

            return result;
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

        la::vector<double> softmax(la::vector<double> const& v)
        {
            la::vector<double> result;
            result.resize(v.size());

            double logZ = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < v.size(); ++j) {
                logZ = ebt::log_add(logZ, v(j));
            }

            for (int i = 0; i < v.size(); ++i) {
                result(i) = std::exp(v(i) - logZ);
            }

            return result;
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

        la::vector<double> logsoftmax(la::vector<double> const& v)
        {
            la::vector<double> result;
            result.resize(v.size());

            double logZ = -std::numeric_limits<double>::infinity();

            for (int j = 0; j < v.size(); ++j) {
                logZ = ebt::log_add(logZ, v(j));
            }

            for (int i = 0; i < v.size(); ++i) {
                result(i) = v(i) - logZ;
            }

            return result;
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
