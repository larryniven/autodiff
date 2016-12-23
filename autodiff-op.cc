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
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = 1 / (1 + std::exp(-v_data[i]));
            }
        }

        void ilogistic_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * output_data[i] * (1 - output_data[i]);
            }
        }

        void relu(la::tensor_like<double>& u, la::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = std::max(0.0, v_data[i]);
            }
        }

        void irelu_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += (output_data[i] > 0 ? grad_data[i] : 0);
            }
        }

        void tanh(la::tensor_like<double>& u, la::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                if (v_data[i] > 0) {
                    double z = std::exp(-2 * v_data[i]);
                    u_data[i] = (1 - z) / (1 + z);
                } else {
                    double z = std::exp(2 * v_data[i]);
                    u_data[i] = (z - 1) / (z + 1);
                }
            }
        }

        void itanh_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * (1 - output_data[i] * output_data[i]);
            }
        }

        void softmax(la::tensor_like<double>& u, la::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double logZ = -std::numeric_limits<double>::infinity();

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int j = 0; j < v.vec_size(); ++j) {
                logZ = ebt::log_add(logZ, v_data[j]);
            }

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = std::exp(v_data[i] - logZ);
            }
        }

        void isoftmax_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double mu = la::dot(grad, output);

            double *result_data = result.data();
            double const *output_data = output.data();
            double const *grad_data = grad.data();

            for (int i = 0; i < grad.vec_size(); ++i) {
                result_data[i] += output_data[i] * (grad_data[i] - mu);
            }
        }

        void logsoftmax(la::tensor_like<double>& u, la::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double logZ = -std::numeric_limits<double>::infinity();

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int j = 0; j < v.vec_size(); ++j) {
                logZ = ebt::log_add(logZ, v_data[j]);
            }

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = v_data[i] - logZ;
            }
        }

        void ilogsoftmax_grad(la::tensor_like<double>& result,
            la::tensor_like<double> const& grad,
            la::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double mu = 0;

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < grad.vec_size(); ++i) {
                mu += grad_data[i];
            }

            for (int i = 0; i < grad.vec_size(); ++i) {
                result_data[i] += grad_data[i] - std::exp(output_data[i]) * mu;
            }
        }

        void corr_linearize(la::tensor_like<double>& result,
            la::tensor_like<double> const& u,
            int f1, int f2)
        {
            assert(u.dim() >= 2);

            unsigned int d3 = u.vec_size() / (u.size(0) * u.size(1));

            assert(result.dim() == 2);
            assert(result.size(0) == u.size(0) * u.size(1));
            assert(result.size(1) == f1 * f2 * d3);

            la::weak_tensor<double> u3 { const_cast<double*>(u.data()), { u.size(0), u.size(1), d3 } };

            int m = 0;

            for (int i = 0; i < u3.size(0); ++i) {
                for (int j = 0; j < u3.size(1); ++j) {

                    int n = 0;

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            for (int k = 0; k < u3.size(2); ++k) {
                                if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                        || i + a - (f1 / 2) >= u3.size(0) || j + b - (f2 / 2) >= u3.size(1)) {
                                    result({m, n}) = 0;
                                } else {
                                    result({m, n}) = u3({i + a - (f1 / 2), j + b - (f2 / 2), k});
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
            la::tensor_like<double> const& u,
            int f1, int f2)
        {
            assert(result.dim() >= 3);

            unsigned int d3 = result.vec_size() / (result.size(0) * result.size(1));

            assert(u.dim() == 2);
            assert(u.size(0) == result.size(0) * result.size(1));
            assert(u.size(1) == f1 * f2 * d3);

            la::weak_tensor<double> result3 { result.data(), { result.size(0), result.size(1), d3 } };

            int m = 0;

            for (int i = 0; i < result3.size(0); ++i) {
                for (int j = 0; j < result3.size(1); ++j) {

                    int n = 0;

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            for (int k = 0; k < result3.size(2); ++k) {
                                if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                        || i + a - (f1 / 2) >= result3.size(0) || j + b - (f2 / 2) >= result3.size(1)) {
                                    // do nothing
                                } else {
                                    result({i + a - (f1 / 2), j + b - (f2 / 2), k}) += u({m, n});
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
