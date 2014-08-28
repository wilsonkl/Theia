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

#include "statx/distributions/evd/gumbel_mle.h"
#include <cmath>
#include <optimo/solvers/newton.h>
#include "statx/utils/common_funcs.h"
#include <glog/logging.h>

namespace statx {
namespace distributions {
namespace evd {
using Eigen::Matrix;
using Eigen::Dynamic;
using std::vector;
// Implementation for xi = 0
// The GEV for the Gumbel case is convex. Thus we solve it with a Newton method.
// We solve then for mu and sigma.
class GumbelMLEProblem : public optimo::Problem<double, 0, 2> {
 public:
  // Constructor
  GumbelMLEProblem(const GumbelMLEObjective& obj,
                   const GumbelMLEGradientFunctor& g,
                   const GumbelMLEHessianFunctor& h) :
      optimo::Problem<double, 0, 2>(obj, g, h) { }
  // Destructor
  virtual ~GumbelMLEProblem(void) { }
};

// The MLE objective function for the Gumbel case is:
// l = m \log(\sigma) + \sum_{i=1}^m \frac{z_i - \mu}{\sigma} +
//     \sum_{i=1}^m \exp{-\frac{z_i - \mu}{\sigma}}
// where m is the number of samples (data).
double GumbelMLEObjective::operator()(const Matrix<double, 2, 1>& x) const {
  const int m = data_.size();
  const double& mu = x(0);
  const double& sigma = x(1);
  const double sigma_inv = 1.0/sigma;
  double acc_1 = 0.0;
  double acc_2 = 0.0;

  for (double z : data_) {
    const double z_minus_mu = z - mu;
    const double exp_arg = z_minus_mu*sigma_inv;
    acc_1 += exp_arg;
    acc_2 += exp(-exp_arg);
  }

  return m*log(sigma) + acc_1 + acc_2;
}

// The Hessian matrix is calculated as follows:
// H = [d^2 l/dmu^2  d^2 l/d\mu d\sigma;
//      d^2 l/d\mu d\sigma  d^2 l / d\sigma^2]
// d^2 l/d\mu^2 = \frac{1}{sigma^2} \sum_{i=1}^m \exp{-\frac{z_i - \mu}{\sigma}}
// d^2 l/d\mu dl = \frac{m}{\sigma^2} -
//  \frac{1}{\sigma^2} \sum_{i=1}^m \exp{-\frac{z_i - \mu}{\sigma}} +
//  \frac{1}{\sigma^3} \sum_{i=1}^m (z_i - \mu)\exp{-\frac{z_i - \mu}{\sigma}}
// d^2 l/d\sigma^2 = -\frac{m}{\sigma^2} +
//  \frac{2}{\sigma^3} \sum_{i=1}^m (z_i - \mu) -
//  \frac{2}{\sigma^3} \sum_{i=1}^m (z_i - \mu)\exp{-\frac{z_i - \mu}{\sigma}} +
//  \frac{1}{\sigma^4} \sum_{i=1}^m (z_i - \mu)^2exp{-\frac{z_i - \mu}{\sigma}}
// The Gumbel MLE problem is convex. Thus the Hessian matrix is positive
// semi-definite, and prevents \sigma to be close to zero.
void GumbelMLEHessianFunctor::operator()(const Matrix<double, 2, 1>& x,
                                         Matrix<double, 2, 2>* h) const {
  const double& mu = x(0);
  const double& sigma = x(1);
  const double sigma_sqrd = sigma*sigma;
  const double sigma_thrd = sigma_sqrd*sigma;
  const double sigma_fth = sigma_thrd*sigma;
  const double sigma_inv = 1.0/sigma;
  const double sigma_sqrd_inv = 1.0/sigma_sqrd;
  const double sigma_thrd_inv = 1.0/sigma_thrd;
  const double sigma_fth_inv = 1.0/sigma_fth;
  const int m = data_.size();

  double acc1 = 0.0;
  double acc2 = 0.0;
  double acc3 = 0.0;
  double acc4 = 0.0f;
  for (double z : data_) {
    const double z_minus_mu = z - mu;
    const double exp_arg = z_minus_mu*sigma_inv;
    const double exp_term = exp(-exp_arg);
    const double exp_term_2 = exp_term*z_minus_mu;
    acc1 += exp_term;  // for d^2lm/dmu^2 & d^2lm/dsigma dmu
    acc2 += exp_term_2;  // for d^2lm/dsigma dmu
    acc3 += z_minus_mu;  // for d^2lm/dsigma^2
    acc4 += z_minus_mu*z_minus_mu*exp_term;  // for d^2lm/dsigma^2
  }
  // Building the Hessian matrix
  (*h)(0, 0) = acc1*sigma_sqrd_inv;
  (*h)(0, 1) = m*sigma_sqrd_inv - (*h)(0, 0) + sigma_thrd_inv*acc2;
  (*h)(1, 0) = (*h)(0, 1);
  (*h)(1, 1) = -m*sigma_sqrd_inv + 2*sigma_thrd_inv*acc3 - 2*sigma_thrd_inv*acc2
      + sigma_fth_inv*acc4;
}

// The gradient is calculated as follows:
// g = [dl/d\mu dl/d\sigma]'
// dl/d\mu = -\frac{m}{\sigma} +
//  \frac{1}{\sigma} \sum_{i=1}^m \exp{-\frac{z_i - \mu}{\sigma}}
// dl/d\sigma = \frac{m}{\sigma} - \frac{1}{\sigma^2} \sum_{i=1}^m (z_i - \mu) +
// \frac{1}{\sigma^2} \sum_{i=1}^m (z_i - \mu)\exp{-\frac{z_i - \mu}{\sigma}}
void GumbelMLEGradientFunctor::operator()(const Matrix<double, 2, 1>& x,
                                          Matrix<double, 2, 1>* g) const {
  const double& mu = x(0);
  const double& sigma = x(1);
  const double sigma_sqrd = sigma*sigma;
  const double sigma_inv = 1.0/sigma;
  const double sigma_sqrd_inv = 1.0/sigma_sqrd;
  const int m = data_.size();
  const double m_over_sigma = m*sigma_inv;
  double acc_mu = 0.0;
  double acc_sigma_1 = 0.0;
  double acc_sigma_2 = 0.0;

  for (double z : data_) {
    const double z_minus_mu = z - mu;
    const double exp_term = exp(-z_minus_mu*sigma_inv);
    acc_mu += exp_term;
    acc_sigma_1 += z_minus_mu;
    acc_sigma_2 += z_minus_mu*exp_term;
  }

  (*g)(0) = sigma_inv*acc_mu - m_over_sigma;  // dL/dmu
  (*g)(1) = m_over_sigma - sigma_sqrd_inv*acc_sigma_1 +  // dL/dsigma
      sigma_sqrd_inv*acc_sigma_2;
}

bool gumbelfit_mle(const vector<double>& data,
                   double* mu,
                   double* sigma) {
  Matrix<double, 2, 1> x;
  // Initial Values computed as in Gumbel function from R's package EVIR.
  // sigma0 <- sqrt(6 * var(data))/pi
  // mu0 <- mean(data) - 0.57722 * sigma0
  const double mean = statx::utils::mean(data);
  const double stddev = statx::utils::stddev(data, mean);
  const double var = stddev*stddev;
  const double sigma0 = sqrt(6 * var)/M_PI;  // sigma
  const double mu0 = mean - 0.57722*sigma0;
  x(1) = sigma0;  // sigma
  x(0) = mu0;  // mu
  CHECK_GT(x(1), 0.0) << "Invalid initial sigma: " << x(1);

  double min_val;
  GumbelMLEObjective gumbel_mle(data);
  GumbelMLEGradientFunctor gumbel_gradient(data);
  GumbelMLEHessianFunctor gumbel_hessian(data);
  GumbelMLEProblem gumbel_mle_problem(gumbel_mle,
                                      gumbel_gradient,
                                      gumbel_hessian);
  optimo::solvers::Newton<double, 0, 2> newton;  // Newton
  auto res = newton(gumbel_mle_problem, &x, &min_val);
  *mu = x(0);
  *sigma = x(1);
  bool exit_flag = res == 0;
  LOG_IF(INFO, !exit_flag) << "MLE did not converge: res=" << res;
  VLOG_IF(1, !exit_flag) << "Values: " << x.transpose()
                         << " min_val: " << min_val;
  return exit_flag;
}
}  // evd
}  // distributions
}  // statx
