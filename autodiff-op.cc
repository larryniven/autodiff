#include "autodiff/autodiff-op.h"
#include "ebt/ebt.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <cblas.h>

namespace autodiff {

    namespace op {

        void logistic(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = 1 / (1 + std::exp(-v_data[i]));
            }
        }

        void ilogistic_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * output_data[i] * (1 - output_data[i]);
            }
        }

        void relu(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            double *u_data = u.data();
            double const *v_data = v.data();

            for (int i = 0; i < v.vec_size(); ++i) {
                u_data[i] = std::max(0.0, v_data[i]);
            }
        }

        void irelu_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += (output_data[i] > 0 ? grad_data[i] : 0);
            }
        }

        void tanh(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
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

        void itanh_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double *result_data = result.data();
            double const *grad_data = grad.data();
            double const *output_data = output.data();

            for (int i = 0; i < output.vec_size(); ++i) {
                result_data[i] += grad_data[i] * (1 - output_data[i] * output_data[i]);
            }
        }

        void softmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
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

        void isoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            double mu = la::cpu::dot(grad, output);

            double *result_data = result.data();
            double const *output_data = output.data();
            double const *grad_data = grad.data();

            for (int i = 0; i < grad.vec_size(); ++i) {
                result_data[i] += output_data[i] * (grad_data[i] - mu);
            }
        }

        void logsoftmax(la::cpu::tensor_like<double>& u, la::cpu::tensor_like<double> const& v)
        {
            assert(u.vec_size() == v.vec_size());

            la::cpu::matrix_like<double> const& m = v.as_matrix();

            la::cpu::vector<double> logZs;
            logZs.resize(m.rows(), -std::numeric_limits<double>::infinity());

            for (int i = 0; i < m.rows(); ++i) {
                for (int j = 0; j < m.cols(); ++j) {
                    logZs(i) = ebt::log_add(logZs(i), m(i, j));
                }
            }

            la::cpu::weak_matrix<double> result {u.data(), m.rows(), m.cols()};

            for (int i = 0; i < result.rows(); ++i) {
                for (int j = 0; j < result.cols(); ++j) {
                    result(i, j) = m(i, j) - logZs(i);
                }
            }
        }

        void ilogsoftmax_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& grad,
            la::cpu::tensor_like<double> const& output)
        {
            assert(result.vec_size() == grad.vec_size() && grad.vec_size() == output.vec_size());

            la::cpu::matrix_like<double> const& grad_m = grad.as_matrix();

            la::cpu::weak_matrix<double> result_m {result.data(),
                grad_m.rows(), grad_m.cols()};

            la::cpu::weak_matrix<double> output_m {const_cast<double*>(output.data()),
                grad_m.rows(), grad_m.cols()};

            la::cpu::vector<double> mu;
            mu.resize(grad_m.rows(), 0);

            la::cpu::vector<double> one;
            one.resize(grad_m.cols(), 1);

            la::cpu::mul(mu, grad_m, one);

            for (int i = 0; i < grad_m.rows(); ++i) {
                for (int j = 0; j < grad_m.cols(); ++j) {
                    result_m(i, j) += grad_m(i, j) - std::exp(output_m(i, j)) * mu(i);
                }
            }
        }

        void corr_linearize(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& u,
            int f1, int f2)
        {
            assert(u.dim() >= 2);

            unsigned int d3 = u.vec_size() / (u.size(0) * u.size(1));

            la::cpu::weak_tensor<double> u3 { const_cast<double*>(u.data()), { u.size(0), u.size(1), d3 } };

            int z = 0;

            double *result_data = result.data();
            double const *u3_data = u3.data();

            unsigned int s0 = u3.size(0);
            unsigned int s1 = u3.size(1);
            unsigned int s2 = u3.size(2);

            for (int i = 0; i < s0; ++i) {
                for (int j = 0; j < s1; ++j) {

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            int base = (i + a - (f1 / 2)) * s1 * s2 + (j + b - (f2 / 2)) * s2;

                            if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                    || i + a - (f1 / 2) >= s0 || j + b - (f2 / 2) >= s1) {
                                // do nothing
                                z += s2;
                            } else {
                                for (int k = 0; k < s2; ++k) {
                                    result_data[z] = u3_data[base + k];
                                    ++z;
                                }
                            }
                        }
                    }

                }
            }
        }

        void corr_linearize_grad(la::cpu::tensor_like<double>& result,
            la::cpu::tensor_like<double> const& u,
            int f1, int f2)
        {
            assert(result.dim() >= 3);

            unsigned int d3 = result.vec_size() / (result.size(0) * result.size(1));

            la::cpu::weak_tensor<double> result3 { result.data(), { result.size(0), result.size(1), d3 } };

            int z = 0;

            double *result3_data = result3.data();
            double const *u_data = u.data();

            unsigned int s0 = result3.size(0);
            unsigned int s1 = result3.size(1);
            unsigned int s2 = result3.size(2);

            for (int i = 0; i < s0; ++i) {
                for (int j = 0; j < s1; ++j) {

                    for (int a = 0; a < f1; ++a) {
                        for (int b = 0; b < f2; ++b) {
                            int base = (i + a - (f1 / 2)) * s1 * s2 + (j + b - (f2 / 2)) * s2;

                            if (i + a - (f1 / 2) < 0 || j + b - (f2 / 2) < 0
                                    || i + a - (f1 / 2) >= s0 || j + b - (f2 / 2) >= s1) {
                                // do nothing
                                z += s2;
                            } else {
                                for (int k = 0; k < s2; ++k) {
                                    result3_data[base + k] += u_data[z];
                                    ++z;
                                }
                            }
                        }
                    }

                }
            }
        }
    }
}
