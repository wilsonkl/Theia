// Copyright (C) 2013  Victor Fragoso <vfragoso@cs.ucsb.edu>
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

#include <glog/logging.h>
#include "gtest/gtest.h"
#include "statx/distributions/evd/gumbel.h"
#include "statx/distributions/evd/gumbel_mle.h"
#include "statx/utils/common_funcs.h"
#include "statx/utils/ecdf.h"

namespace statx {
namespace distributions {
namespace evd {
using std::vector;
using Eigen::Matrix;

// Generated w/ Matlab gumbel (minima) params: mu=10, sigma=2.5
const vector<double> gumbel_data {
  -1.032351e+01, -4.245950e+00, -8.203873e+00, -1.355269e+01, -1.038544e+01,
      -1.124625e+01, -1.063522e+01, -1.321805e+01, -5.789039e+00, -6.715748e+00,
      -7.204733e+00, -5.914938e+00, -6.712514e+00, -1.187329e+01, -9.888159e+00,
      -1.318722e+01, -1.473351e+01, -1.400459e+01, -1.144865e+01, -1.159702e+01,
      -8.971152e+00, -9.652318e+00, -8.370906e+00, -1.062457e+01, -8.378861e+00,
      -8.440498e+00, -1.186328e+01, -1.233382e+01, -1.308864e+01, -1.144545e+01,
      -8.907367e+00, -5.609869e+00, -1.041976e+01, -1.125896e+01, -1.251216e+01,
      -2.771665e+00, -1.085445e+01, -8.109022e+00, -1.113820e+01, -1.087643e+01,
      -6.960764e+00, -1.154858e+01, -5.907846e+00, -1.215154e+01, -2.705683e-01,
      -7.865772e+00, -1.057456e+01, -5.128051e+00, -1.313327e+01, -6.756149e+00,
      -4.473102e+00, -8.164309e+00, -7.209097e+00, -9.340718e+00, -6.780737e+00,
      -1.399431e+01, -1.124063e+01, -1.218027e+01, -1.144742e+01, -2.944992e+00,
      -5.467856e+00, -1.171494e+01, -7.797793e+00, -1.295230e+01, -1.161703e+01,
      -1.130986e+01, -7.935839e+00, -5.336572e+00, -5.395872e+00, -9.126612e+00,
      -6.507492e+00, -1.064095e+01, -1.484286e+00, 1.637005e+00, -1.226217e+01,
      -1.025930e+01, -1.068583e+01, -1.106780e+01, -6.682201e+00, -1.001920e+01,
      -1.156577e+01, -1.147179e+01, -4.543793e+00, -2.782129e+00, -8.956695e+00,
      -1.346377e+01, -9.649415e+00, -1.185846e+01, -2.477960e+00, -1.067252e+01,
      -4.536605e+00, -4.498064e+00, -6.575974e+00, -4.131232e+00, -1.352351e+01,
      -2.048649e+00, -1.237615e+01, -1.056744e+01, -9.168998e+00, -1.222745e+00
      };

// Generated w/ Matlab gumbel params: mu=-1.0, sigma=2.0
const vector<double> gumbel_data2 {
  -4.170405e+00, -5.626369e+00, 4.489738e-01, -5.802430e+00, -2.560473e+00,
      6.895796e-01, -5.088696e-01, -2.009942e+00, -7.273559e+00, -7.662819e+00,
      2.277881e-01, -8.023256e+00, -7.257273e+00, -1.649156e+00, -4.003024e+00,
      3.384557e-01, -1.293949e+00, -5.860208e+00, -3.914016e+00, -7.371324e+00,
      -2.725548e+00, 1.407312e+00, -4.621349e+00, -6.368099e+00, -2.895954e+00,
      -3.564483e+00, -3.428848e+00, -1.132462e+00, -2.723649e+00, 1.363020e-01,
      -3.110676e+00, 1.475157e+00, -5.000149e-01, 1.246867e+00, 6.931838e-01,
      -4.277291e+00, -3.020708e+00, -7.230173e-01, -6.949524e+00, 1.428852e+00,
      -1.387562e+00, -1.074384e+00, -3.639485e+00, -3.946654e+00, 3.440429e-02,
      -1.674219e+00, -1.425484e+00, -2.658065e+00, -3.137804e+00, -3.535582e+00,
      -4.949626e-01, -2.903319e+00, -2.720906e+00, 1.937019e-01, 5.109771e-01,
      -1.723592e+00, -7.384042e+00, -8.503866e-01, -2.248415e+00, -1.931417e-01,
      -3.503568e+00, -3.760525e-01, -1.767495e+00, -3.054474e+00, -5.316633e+00,
      -7.361217e+00, -2.011966e+00, 3.621359e-01, 2.856413e-01, -3.898843e-01,
      -4.503157e+00, -3.713851e-01, -4.165153e+00, -3.092269e-01, -6.224676e+00,
      -9.026701e-01, -2.700599e-02, -3.529824e-01, -2.449541e+00, -1.580573e+00,
      -9.117904e-01, -4.371216e+00, -2.248391e+00, -2.027193e+00, -5.896687e+00,
      -5.499940e-01, -3.559351e+00, -3.526581e+00, -1.068331e+00, -2.138506e+00,
      8.947577e-01, 1.142959e+00, -1.913387e+00, -3.776356e+00, -6.368646e+00,
      4.268235e-01, -2.144747e+00, -1.558582e+00, 1.977270e+00, -8.325893e-01 };

// Gumbel pdfdata mu=0.0, sigma=1.0 generated w/ Matlab
// Note: Matlab generates Gumbel data simulating minima!
const vector<double> gumbelpdf_data {
  4.566281e-03, 8.346655e-03, 1.426926e-02, 2.296125e-02, 3.497812e-02,
      5.070711e-02, 7.028478e-02, 9.354650e-02, 1.200176e-01, 1.489468e-01,
      1.793741e-01, 2.102195e-01, 2.403784e-01, 2.688094e-01, 2.946053e-01,
      3.170419e-01, 3.356036e-01, 3.499872e-01, 3.600895e-01, 3.659821e-01,
      3.678794e-01, 3.661042e-01, 3.610529e-01, 3.531656e-01, 3.428988e-01,
      3.307043e-01, 3.170133e-01, 3.022245e-01, 2.866971e-01, 2.707472e-01,
      2.546464e-01, 2.386228e-01, 2.228639e-01, 2.075191e-01, 1.927046e-01,
      1.785065e-01, 1.649857e-01, 1.521812e-01, 1.401140e-01, 1.287904e-01,
      1.182050e-01, 1.083426e-01, 9.918156e-02, 9.069447e-02, 8.285046e-02,
      7.561618e-02, 6.895690e-02, 6.283736e-02, 5.722239e-02, 5.207745e-02,
      4.736901e-02, 4.306481e-02, 3.913406e-02, 3.554758e-02, 3.227787e-02,
      2.929913e-02, 2.658724e-02, 2.411977e-02, 2.187588e-02, 1.983630e-02,
      1.798323e-02, 1.630029e-02, 1.477239e-02, 1.338570e-02, 1.212753e-02,
      1.098627e-02, 9.951302e-03, 9.012928e-03, 8.162296e-03, 7.391337e-03,
      6.692700e-03 };

// Gumbel cdfdata mu=0.0, sigma=1.0
// Note: Matlab generates Gumbel data simulating minima!
const vector<double> gumbelcdf_data {
  6.179790e-04, 1.248398e-03, 2.358693e-03, 4.194642e-03, 7.061962e-03,
      1.131429e-02, 1.733201e-02, 2.549439e-02, 3.614860e-02, 4.958009e-02,
      6.598804e-02, 8.546887e-02, 1.080090e-01, 1.334868e-01, 1.616828e-01,
      1.922956e-01, 2.249618e-01, 2.592769e-01, 2.948163e-01, 3.311543e-01,
      3.678794e-01, 4.046077e-01, 4.409910e-01, 4.767237e-01, 5.115448e-01,
      5.452392e-01, 5.776358e-01, 6.086053e-01, 6.380562e-01, 6.659307e-01,
      6.922006e-01, 7.168626e-01, 7.399341e-01, 7.614492e-01, 7.814556e-01,
      8.000107e-01, 8.171795e-01, 8.330317e-01, 8.476403e-01, 8.610793e-01,
      8.734230e-01, 8.847445e-01, 8.951149e-01, 9.046032e-01, 9.132753e-01,
      9.211937e-01, 9.284177e-01, 9.350030e-01, 9.410020e-01, 9.464632e-01,
      9.514320e-01, 9.559504e-01, 9.600574e-01, 9.637887e-01, 9.671775e-01,
      9.702540e-01, 9.730462e-01, 9.755796e-01, 9.778776e-01, 9.799616e-01,
      9.818511e-01, 9.835639e-01, 9.851163e-01, 9.865231e-01, 9.877977e-01,
      9.889525e-01, 9.899985e-01, 9.909460e-01, 9.918040e-01, 9.925811e-01,
      9.932847e-01 };

TEST(Gumbel, MLE_Objective) {
  GumbelMLEObjective mle(gumbel_data);
  Matrix<double, 2, 1> x;

  // True Values
  x(0) = 10.0;  // mu (Location parameter)
  x(1) = 2.5;  // sigma (Scale parameter)
  double mle_val = mle(x);
  VLOG(1) << "MLE Gumbel=" << mle_val;
  ASSERT_GT(mle_val, 0.0);

  // Gumbel data 2
  GumbelMLEObjective mle2(gumbel_data2);

  // True Values
  x(0) = 1.0;  // mu (Location parameter)
  x(1) = 2.0;  // sigma (Scale parameter)
  mle_val = mle(x);
  VLOG(1) << "MLE Gumbel=" << mle_val;
  ASSERT_GT(mle_val, 0.0);
}

TEST(Gumbel, Hessian1) {
  // Case 1
  GumbelMLEHessianFunctor gumbel_hessian(gumbel_data);
  Matrix<double, 2, 2> h;
  Matrix<double, 2, 1> x;
  Matrix<double, 2, 2> L;
  x(0) = -10.0;
  x(1) = 2.5;
  gumbel_hessian(x, &h);
  L(0, 0) = 4.5212;
  L(0, 1) = -4.2175;
  L(1, 0) = 0;
  L(1, 1) = 5.7530;
  // Checking PositiveDefinite matrix via Cholesky decomposition
  Matrix<double, 2, 2> res = L.transpose()*L - h;
  double residual_norm = res.norm()/4;
  VLOG(1) << "Hessian Matrix(Gumbel): \n" << h;
  VLOG(1) << "Residual Matrix L'*L - h(Gumbel) norm: " << residual_norm;
  ASSERT_LT(residual_norm, 1.0);
}

TEST(Gumbel, Hessian2) {
  // Case 2
  Matrix<double, 2, 2> h;
  Matrix<double, 2, 1> x;
  Matrix<double, 2, 2> L;
  GumbelMLEHessianFunctor gumbel_hessian2(gumbel_data2);
  x(0) = -1.0;
  x(1) = 2.0;
  gumbel_hessian2(x, &h);
  L(0, 0) = 10.5997;
  L(0, 1) = -32.3732;
  L(1, 0) = 0;
  L(1, 1) = 11.2595;
  Matrix<double, 2, 2> res = L.transpose()*L - h;
  double residual_norm = res.norm()/4;
  VLOG(1) << "Hessian Matrix(Gumbel 2): \n" << h;
  VLOG(1) << "Residual Matrix L'*L - h(Gumbel 2) norm: " << residual_norm;
  ASSERT_LT(residual_norm, 1.0);
}

TEST(Gumbel, Gradient_Gumbel_ZeroNorm) {
  GumbelMLEGradientFunctor gradient(gumbel_data);
  Matrix<double, 2, 1> x, g;
  // ML (Newton)
  x(0) = -10.4974;  // mu (Location parameter)
  x(1) = 2.76127;  // sigma (Scale parameter)
  gradient(x, &g);
  VLOG(1) << "Gradient: " << g.transpose();
  ASSERT_NEAR(0.0, g.norm(), 1e-3);  
}

TEST(Gumbel, FitMLE1) {
  // Case 1
  const double mu_gt = -10.0;
  const double sigma_gt = 2.5;
  double mu = -10.0;
  double sigma = 2.5;
  ASSERT_TRUE(gumbelfit(gumbel_data, &mu, &sigma));
  VLOG(1) << "mu=" << mu << " sigma=" << sigma;
  GumbelMLEGradientFunctor gradient(gumbel_data);
  Matrix<double, 2, 1> x, g;
  // ML computed from Gradient Descent
  x(0) = mu;  // mu (Location parameter)
  x(1) = sigma;  // sigma (Scale parameter)
  gradient(x, &g);
  VLOG(1) << "Gradient: " << g.transpose();
  ASSERT_NEAR(0.0, g.norm(), 1e-3);
  EXPECT_NEAR(mu_gt, mu, 5.0);
  EXPECT_NEAR(sigma_gt, sigma, 5.0);
}

TEST(Gumbel, FitMLE2) {
  // Case 2
  const double mu_gt = -1.0;
  const double sigma_gt = 2.0;
  double mu = 10.0;
  double sigma = 2.5;
  Matrix<double, 2, 1> x, g;
  ASSERT_TRUE(gumbelfit(gumbel_data2, &mu, &sigma));
  VLOG(1) << "mu=" << mu << " sigma=" << sigma;
  GumbelMLEGradientFunctor gradient2(gumbel_data2);
  // ML computed from Gradient Descent
  x(0) = mu;  // mu (Location parameter)
  x(1) = sigma;  // sigma (Scale parameter)
  gradient2(x, &g);
  VLOG(1) << "Gradient: " << g.transpose();
  ASSERT_NEAR(0.0, g.norm(), 1e-3);
  EXPECT_NEAR(mu_gt, mu, 5.0);
  EXPECT_NEAR(sigma_gt, sigma, 5.0);
}

// TODO(vfragoso): Test with data generated from Gumbel for maxima!
TEST(Gumbel, PDF) {
  // Params
  const double mu = 0.0;
  const double sigma = 1.0;
  // Domain
  const int nsamples = gumbelpdf_data.size();
  double x = -2.0;
  const double dx = 0.1;
  // Testing
  for (int i = 0; i < nsamples; i++) {
    double y_gt = gumbelpdf_data[i];
    double y = gumbelpdf(x, mu, sigma);
    ASSERT_NEAR(y_gt, y, 1e-4);
    x += dx;
  }
}

TEST(Gumbel, CDF) {
  // Params
  const double mu = 0.0;
  const double sigma = 1.0;
  // Domain
  const int nsamples = gumbelcdf_data.size();
  double x = -2.0;
  const double dx = 0.1;
  // Testing
  for (int i = 0; i < nsamples; i++) {
    double y_gt = gumbelcdf_data[i];
    double y = gumbelcdf(x, mu, sigma);
    ASSERT_NEAR(y_gt, y, 1e-4);
    x += dx;
  }
}
}  // evd
}  // distributions
}  // statx
