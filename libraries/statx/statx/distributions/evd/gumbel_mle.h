// Copyright (C) 2014  Victor Fragoso <vfragoso@cs.ucsb.edu>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of the University of California, Santa Barbara nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL VICTOR FRAGOSO BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef STATX_DISTRIBUTIONS_EVD_GUMBEL_MLE_H_
#define STATX_DISTRIBUTIONS_EVD_GUMBEL_MLE_H_

#include <optimo/core/objects.h>
#include <vector>

namespace statx {
namespace distributions {
namespace evd {
using Eigen::Matrix;
using Eigen::Dynamic;
using std::vector;

// Solve for parameters of Gumbel distribution using the MLE estimator
bool gumbelfit_mle(const vector<double>& data,
                   double* mu,
                   double* sigma);

// General MLE problem for Gumbel, which is a special case of GEV when xi = 0.
class GumbelMLEObjective : public optimo::ObjectiveFunctor<double, 2> {
  // TODO(vfragoso): Implement me!! See if Primal-Dual formulation is better
 public:
  // Constructor
  explicit GumbelMLEObjective(const std::vector<double>& data) :
      data_(data) { }

  // Destructor
  virtual ~GumbelMLEObjective(void) { }

  // Impl
  virtual double operator()(const Eigen::Matrix<double, 2, 1>& x) const;
 protected:
  const std::vector<double>& data_;
};

// The Hessian functor
class GumbelMLEHessianFunctor : public optimo::HessianFunctor<double, 2> {
 public:
  // Constructor
  explicit GumbelMLEHessianFunctor(const std::vector<double>& data) :
      data_(data) { }

  // Destructor
  virtual ~GumbelMLEHessianFunctor(void) { }

  virtual void
  operator()(const Eigen::Matrix<double, 2, 1>& x,
             Eigen::Matrix<double, 2, 2>* h) const;
 protected:
  const std::vector<double>& data_;
};

// The gradient functor
class GumbelMLEGradientFunctor : public optimo::GradientFunctor<double, 2> {
 public:
  // Constructor
  explicit GumbelMLEGradientFunctor(const std::vector<double>& data) :
      data_(data) { }

  // Destructor
  virtual ~GumbelMLEGradientFunctor(void) { }

  // Implementation
  virtual void
  operator()(const Eigen::Matrix<double, 2, 1>& x,
             Eigen::Matrix<double, 2, 1>* g) const;

 protected:
  const std::vector<double>& data_;
};
}  // evd
}  // distributions
}  // statx
#endif  // STATX_DISTRIBUTIONS_EVD_GUMBEL_MLE_H_
