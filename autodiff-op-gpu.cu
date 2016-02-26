#include "autodiff/autodiff-op-gpu.h"
#include "la/la-gpu.h"
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>

namespace autodiff {

    namespace op {

        namespace gpu {

            void iouter_prod(la::gpu::matrix_like<double>& result,
                la::gpu::vector_like<double> const& x,
                la::gpu::vector_like<double> const& y)
            {
                double alpha = 1;
                cublasDger(la::gpu::device::get_handle(), x.size(), y.size(), &alpha,
                    x.data(), 1, y.data(), 1, result.data(), x.size());
            }

            void ilmul(la::gpu::vector_like<double>& result,
                la::gpu::matrix_like<double> const& a,
                la::gpu::vector_like<double> const& x)
            {
                assert(a.rows() == x.size());

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
                    thrust::get<0>(t) = 1 / (1 + exp(-thrust::get<1>(t)));
                }
            };

            void logistic(la::gpu::vector_like<double>& u, la::gpu::vector_like<double> const& v)
            {
                assert(u.size() == v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.begin()),
                        thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.end()),
                        thrust::device_ptr<double const>(v.end()))),
                    ilogistic_op());
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

            void ilogistic_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output)
            {
                assert(grad.size() == output.size());

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

            void relu(la::gpu::vector_like<double>& u, la::gpu::vector_like<double> const& v)
            {
                assert(u.size() == v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.begin()),
                        thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.end()),
                        thrust::device_ptr<double const>(v.end()))),
                    relu_op());
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

            void irelu_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output)
            {
                assert(grad.size() == output.size());

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

                    if (v > 0) {
                        double z = std::exp(-2 * v);
                        result = (1 - z) / (1 + z);
                    } else {
                        double z = std::exp(2 * v);
                        result = (z - 1) / (z + 1);
                    }
                }
            };

            void tanh(la::gpu::vector_like<double>& u, la::gpu::vector_like<double> const& v)
            {
                assert(u.size() == v.size());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.begin()),
                        thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.end()),
                        thrust::device_ptr<double const>(v.end()))),
                    tanh_op());
            }

            struct itanh_grad_op {
                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    auto& result = thrust::get<0>(t);
                    auto& grad = thrust::get<1>(t);
                    auto& output = thrust::get<2>(t);

                    result += grad * (1 - output * output);
                }
            };

            void itanh_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output)
            {
                assert(grad.size() == output.size());

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
                        return a + log(1 + exp(b - a));
                    } else {
                        return b + log(1 + exp(a - b));
                    }
                }
            };

            struct isoftmax_op {
                double s;

                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    thrust::get<0>(t) = exp(thrust::get<1>(t) - s);
                }

            };

            void softmax(la::gpu::vector_like<double>& u,
                la::gpu::vector_like<double> const& v)
            {
                assert(u.size() == v.size());

                double inf = std::numeric_limits<double>::infinity();

                double logZ = thrust::reduce(thrust::device_ptr<double const>(v.begin()),
                    thrust::device_ptr<double const>(v.end()), -inf, log_add_op());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.begin()),
                        thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.end()),
                        thrust::device_ptr<double const>(v.end()))),
                    isoftmax_op { logZ });
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

            void isoftmax_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output)
            {
                assert(grad.size() == output.size());

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

                template <class T>
                __host__ __device__
                void operator()(T t) const
                {
                    thrust::get<0>(t) = thrust::get<1>(t) - s;
                }

            };

            void logsoftmax(la::gpu::vector_like<double>& u,
                la::gpu::vector_like<double> const& v)
            {
                assert(u.size() == v.size());

                double inf = std::numeric_limits<double>::infinity();

                double logZ = thrust::reduce(thrust::device_ptr<double const>(v.begin()),
                    thrust::device_ptr<double const>(v.end()), -inf, log_add_op());

                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.begin()),
                        thrust::device_ptr<double const>(v.begin()))),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        thrust::device_ptr<double>(u.end()),
                        thrust::device_ptr<double const>(v.end()))),
                    ilogsoftmax_op { logZ });
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

                    result += grad - exp(output) * mu;
                }

            };

            void ilogsoftmax_grad(la::gpu::vector_like<double>& result,
                la::gpu::vector_like<double> const& grad,
                la::gpu::vector_like<double> const& output)
            {
                assert(grad.size() == output.size());

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
