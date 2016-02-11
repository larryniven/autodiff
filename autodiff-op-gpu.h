#ifndef AUTODIFF_OP_GPU_H
#define AUTODIFF_OP_GPU_H

#include "la/la-gpu.h"

namespace autodiff {

    namespace op {

        namespace gpu {

            void iouter_prod(la::gpu::matrix<double>& result,
                la::gpu::vector<double> const& x,
                la::gpu::vector<double> const& y);

            void ilmul(la::gpu::vector<double>& result,
                la::gpu::matrix<double> const& a,
                la::gpu::vector<double> const& x);

            void logistic(la::gpu::vector<double>& u,
                la::gpu::vector<double> const& v);

            void ilogistic_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output);

            void relu(la::gpu::vector<double>& u,
                la::gpu::vector<double> const& v);

            void irelu_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output);

            void tanh(la::gpu::vector<double>& u,
                la::gpu::vector<double> const& v);

            void itanh_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output);

            void softmax(la::gpu::vector<double>& u,
                la::gpu::vector<double> const& v);

            void isoftmax_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output);

            void logsoftmax(la::gpu::vector<double>& u,
                la::gpu::vector<double> const& v);

            void ilogsoftmax_grad(la::gpu::vector<double>& result,
                la::gpu::vector<double> const& grad,
                la::gpu::vector<double> const& output);
        }
    }
}

#endif
