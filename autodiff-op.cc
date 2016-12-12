#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <cblas.h>

namespace autodiff {

    namespace op {

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

        void conv_linearize(la::matrix_like<double>& result,
            la::tensor_like<double> const& u,
            int f1, int f2)
        {
            assert(u.dim() == 3);

            assert(result.rows() == u.size(0) * u.size(1));
            assert(result.cols() == f1 * f2 * u.size(2));

            int m = 0;

            for (int i = 0; i < u.size(0); ++i) {
                for (int j = 0; j < u.size(1); ++j) {

                    int n = 0;

                    for (int a = std::max<int>(0, (f1 / 2) - i); a < f1; ++a) {
                        for (int b = std::max<int>(0, (f2 / 2) - j); b < f2; ++b) {
                            for (unsigned int k = 0; k < u.size(2); ++k) {
                                result(m, n) = u({(unsigned int)(i + a), (unsigned int)(j + b), k});
                                ++n;
                            }
                        }
                    }

                    ++m;

                }
            }
        }

        void conv_linearize_grad(la::tensor_like<double>& result,
            la::matrix_like<double> const& u,
            int f1, int f2)
        {
            assert(result.dim() == 3);

            assert(u.rows() == result.size(0) * result.size(1));
            assert(u.cols() == f1 * f2 * result.size(2));

            int m = 0;

            for (int i = 0; i < result.size(0); ++i) {
                for (int j = 0; j < result.size(1); ++j) {

                    int n = 0;

                    for (int a = std::max<int>(0, (f1 / 2) - i); a < f1; ++a) {
                        for (int b = std::max<int>(0, (f2 / 2) - j); b < f2; ++b) {
                            for (unsigned int k = 0; k < result.size(2); ++k) {
                                result({(unsigned int)(i + a), (unsigned int)(j + b), k}) += u(m, n);
                                ++n;
                            }
                        }
                    }

                    ++m;

                }
            }
        }
    }
}
