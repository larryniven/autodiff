#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <cblas.h>

namespace autodiff {

    namespace op {

        void logistic(la::tensor_like<double>& u, la::tensor_like<double> const& v)
        {
            assert(u.size() == v.size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.size(); ++i) {
                u_data[i] = 1 / (1 + std::exp(-v_data[i]));
            }
        }

        void ilogistic_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.size() == grad.size() && grad.size() == output.size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.size(); ++i) {
                result_data[i] += grad_data[i] * output_data[i] * (1 - output_data[i]);
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

        void corr_linearize(la::matrix_like<double>& result,
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

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            for (unsigned int k = 0; k < u.size(2); ++k) {
                                if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                        || i + a - (f1 / 2) >= u.size(0) || j + b - (f2 / 2) >= u.size(1)) {
                                    result(m, n) = 0;
                                } else {
                                    result(m, n) = u({(unsigned int)(i + a - (f1 / 2)), (unsigned int)(j + b - (f2 / 2)), k});
                                }

                                ++n;
                            }
                        }
                    }

                    ++m;

                }
            }
        }

        void corr_linearize_grad(la::tensor_like<double>& result,
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

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            for (unsigned int k = 0; k < result.size(2); ++k) {
                                if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                        || i + a - (f1 / 2) >= result.size(0) || j + b - (f2 / 2) >= result.size(1)) {
                                    // do nothing
                                } else {
                                    result({(unsigned int)(i + a - (f1 / 2)), (unsigned int)(j + b - (f2 / 2)), k}) += u(m, n);
                                }

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
