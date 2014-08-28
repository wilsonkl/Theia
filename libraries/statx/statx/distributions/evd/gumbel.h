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

#ifndef STATX_DISTRIBUTIONS_EVD_GUMBEL_H_
#define STATX_DISTRIBUTIONS_EVD_GUMBEL_H_

#include <cmath>
#include <vector>
#include <limits>
#include "statx/distributions/evd/common.h"
namespace statx {
namespace distributions {
namespace evd {
// Evaluates the Gumbel density function for Maxima
// mu: location
// sigma: scale
inline double gumbelpdf(const double x,
                        const double mu,
                        const double sigma) {
  if (sigma <= 0) return std::numeric_limits<double>::infinity();
  double sigma_inv = 1.0/sigma;
  double arg = (x - mu)*sigma_inv;
  double arg1 = exp(-arg);
  return sigma_inv*exp(-arg - arg1);
}

// Evaluates the Gumbel CDF for Maximq
// mu: location
// sigma: scale
inline double gumbelcdf(const double x,
                        const double mu,
                        const double sigma) {
  if (sigma <= 0) return std::numeric_limits<double>::infinity();
  double arg = (x - mu)/sigma;
  return exp(-exp(-arg));
}

inline double gumbel_quantile(const double p,
                              const double mu,
                              const double sigma) {
  // p \in [0, 1]
  if (p < 0.0 || p > 1.0 || sigma <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  double log_term = -log(p);
  return mu - sigma*log(log_term);
}


// Find the parameters for the Gumbel distribution by solcinf
// the MLE problem given the data. The distribution has two params:
// mu: location
// sigma: scale
bool gumbelfit(const std::vector<double>& data,
               double* mu,
               double* sigma,
               FitType fit_type = MLE);
}  // evd
}  // distributions
}  // statx
#endif  // STATX_DISTRIBUTIONS_EVD_GUMBEL_H_
