#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace autodiff {

    namespace op {

        namespace gpu {

            void iouter_prod(la::gpu::matrix<double>& result,
                la::gpu::vector<double> const& x,
                la::gpu::vector<double> const& y)
            {
                result.resize(x.size(), y.size());
                double alpha = 1;
                cublasDger(la::gpu::device::get_handle(), x.size(), y.size(), &alpha,
                    x.data(), 1, y.data(), 1, result.data(), x.size());
            }

            void ilmult(la::gpu::vector<double>& result,
                la::gpu::matrix<double> const& a,
                la::gpu::vector<double> const& x)
            {
                result.resize(a.cols());
                double alpha = 1;
                double beta = 1;
                cublasDgemv(la::gpu::device::get_handle(), CUBLAS_OP_T, a.rows(), a.cols(),
                    &alpha, a.data(), a.rows(), x.data(), 1, &beta, result.data(), 1);
            }

            struct ilogistic_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    thrust::get<0>(t) = 1 / (1 + std::exp(-thrust::get<1>(t)));
                }
            };

            la::gpu::vector<double> logistic(la::gpu::vector<double> const& v)
            {
                la::gpu::vector<double> result;
                result.resize(v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()), thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()), thrust::device_ptr<double const>(v.end()))),
                    ilogistic_op());

                return result;
            }

            struct ilogistic_grad_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += grad * output * (1 - output);
                }
            };

            void ilogistic_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output)
            {
                assert(grad.size() == output.size());

                result.resize(grad.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()),
                        thrust::device_ptr<double const>(grad.begin()),
                        thrust::device_ptr<double const>(output.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()),
                        thrust::device_ptr<double const>(grad.end()),
                        thrust::device_ptr<double const>(output.end()))),
                    ilogistic_grad_op());
            }

            struct relu_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    thrust::get<0>(t) = (thrust::get<1>(t) > 0 ? thrust::get<1>(t) : 0);
                }
            };

            la::gpu::vector<double> relu(la::gpu::vector<double> const& v)
            {
                la::gpu::vector<double> result;
                result.resize(v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()), thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()), thrust::device_ptr<double const>(v.end()))),
                    relu_op());

                return result;
            }

            struct irelu_grad_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += (output > 0 ? grad : 0);
                }
            };

            void irelu_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output)
            {
                assert(grad.size() == output.size());

                result.resize(grad.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()),
                        thrust::device_ptr<double const>(grad.begin()),
                        thrust::device_ptr<double const>(output.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()),
                        thrust::device_ptr<double const>(grad.end()),
                        thrust::device_ptr<double const>(output.end()))),
                    irelu_grad_op());
            }

            struct tanh_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& v = thrust::get<1>(t);

                    double z1 = std::exp(v);
                    double z2 = std::exp(-v);
                    result = (z1 - z2) / (z1 + z2);
                }
            };

            la::gpu::vector<double> tanh(la::gpu::vector<double> const& v)
            {
                la::gpu::vector<double> result;
                result.resize(v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()), thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()), thrust::device_ptr<double const>(v.end()))),
                    tanh_op());

                return result;
            }

            struct itanh_grad_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += (output > 0 ? grad : 0);
                }
            };

            void itanh_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output)
            {
                assert(grad.size() == output.size());

                result.resize(grad.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()),
                        thrust::device_ptr<double const>(grad.begin()),
                        thrust::device_ptr<double const>(output.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()),
                        thrust::device_ptr<double const>(grad.end()),
                        thrust::device_ptr<double const>(output.end()))),
                    itanh_grad_op());
            }

            struct log_add_op {
                __host__ __device__
                double operator()(double a, double b) const
                {
                    if (a > b) {
                        return b + std::log(1 + std::exp(b - a));
                    } else {
                        return a + std::log(1 + std::exp(a - b));
                    }
                }
            };

            struct isoftmax_op {
                double s;

                __host__ __device__
                void operator()(double& x) const
                {
                    x = std::exp(x - s);
                }

            };

            la::gpu::vector<double> softmax(la::gpu::vector<double> const& v)
            {
                la::gpu::vector<double> result;
                result.resize(v.size());

                double inf = std::numeric_limits<double>::infinity();

                double logZ = thrust::reduce(thrust::device_ptr<double const>(v.begin()),
                    thrust::device_ptr<double const>(v.end()), -inf, log_add_op());

                thrust::for_each(
                    thrust::device_ptr<double>(result.begin()),
                    thrust::device_ptr<double>(result.end()),
                    isoftmax_op { logZ });

                return result;
            }

            struct isoftmax_grad_op {
                double mu;

                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += output * (grad - mu);
                }

            };

            void isoftmax_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output)
            {
                assert(grad.size() == output.size());

                result.resize(grad.size());

                double mu = la::gpu::dot(grad, output);

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()),
                        thrust::device_ptr<double const>(grad.begin()),
                        thrust::device_ptr<double const>(output.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()),
                        thrust::device_ptr<double const>(grad.end()),
                        thrust::device_ptr<double const>(output.end()))),
                    isoftmax_grad_op { mu });
            }

            struct ilogsoftmax_op {
                double s;

                __host__ __device__
                void operator()(double& x) const
                {
                    x -= s;
                }

            };

            la::gpu::vector<double> logsoftmax(la::gpu::vector<double> const& v)
            {
                la::gpu::vector<double> result;
                result.resize(v.size());

                double inf = std::numeric_limits<double>::infinity();

                double logZ = thrust::reduce(thrust::device_ptr<double const>(v.begin()),
                    thrust::device_ptr<double const>(v.end()), -inf, log_add_op());

                thrust::for_each(
                    thrust::device_ptr<double>(result.begin()),
                    thrust::device_ptr<double>(result.end()),
                    ilogsoftmax_op { logZ });

                return result;
            }

            struct ilogsoftmax_grad_op {
                double mu;

                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += grad - std::exp(output) * mu;
                }

            };

            void ilogsoftmax_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output)
            {
                assert(grad.size() == output.size());

                result.resize(grad.size());

                double mu = thrust::reduce(
                    thrust::device_ptr<double const>(grad.begin()),
                    thrust::device_ptr<double const>(grad.end()),
                    0.0, thrust::plus<double>());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.begin()),
                        thrust::device_ptr<double const>(grad.begin()),
                        thrust::device_ptr<double const>(output.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(result.end()),
                        thrust::device_ptr<double const>(grad.end()),
                        thrust::device_ptr<double const>(output.end()))),
                    ilogsoftmax_grad_op { mu });
            }

        }
    }
}
