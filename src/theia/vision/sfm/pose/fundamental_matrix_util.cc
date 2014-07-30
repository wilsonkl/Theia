// Copyright (C) 2014 The Regents of the University of California (Regents).
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
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "theia/vision/sfm/pose/fundamental_matrix_util.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace theia {

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {

inline Matrix3d CreateCMatrix(const Matrix3d& a, const Matrix3d& b) {
  const double a1_b3 = a.col(0).dot(b.col(2));
  const double a2_b3 = a.col(1).dot(b.col(2));
  Matrix3d cmatrix;
  cmatrix <<
      a1_b3 * a1_b3, a(2, 0) * a(2, 0) + a(2, 2) * a(2, 2), 1,
      a2_b3 * a1_b3, a(2, 0) * a(2, 1), 0,
      a2_b3 * a2_b3, a(2, 1) * a(2, 1) + a(2, 2) * a(2, 2), 1;
  return cmatrix;
}

}  // namespace

// Given a fundmental matrix, decompose the fundmental matrix and recover focal
// lengths f1, f2 >0 such that diag([f2 f2 1]) F diag[f1 f1 1]) is a valid
// essential matrix. This assumes a principal point of (0, 0) for both cameras.
//
// This code is based off of the paper "Recovering Unknown Focal Lengths in
// Self-Calibration: An Essentially Linear Algorithm and Degenerate
// Configurations" by Newsam et al, International Archives of Photogrammetry and
// Remote Sensing (1996).
//
// Returns true on success, false otherwise.
bool FocalLengthsFromFundamentalMatrix(const double fmatrix[3 * 3],
                                       double* focal_length1,
                                       double* focal_length2) {
  Eigen::Map<const Eigen::Matrix3d> fmatrix_map(fmatrix);
  Eigen::JacobiSVD<Matrix3d> svd(fmatrix_map,
                                 Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Matrix3d& matrix_u = svd.matrixU();
  const Vector3d& singular_values = svd.singularValues();

  if (sqrt(fabs(singular_values(0))) == 0.0 ||
      sqrt(fabs(singular_values(1))) == 0.0) {
    return false;
  }

  const Matrix3d lhs = CreateCMatrix(matrix_u, fmatrix_map);
  const Vector3d rhs(singular_values(0) * singular_values(0),
                     0,
                     singular_values(1) * singular_values(1));

  const Vector3d omegas = lhs.fullPivLu().solve(rhs);

  const double focal_length1_sq = 1.0 / (1.0 - omegas(0));
  const double focal_length2_sq = 1.0 + omegas(1) / omegas(2);

  if (focal_length1_sq < 0 || focal_length2_sq < 0) {
    return false;
  }

  *focal_length1 = sqrt(focal_length1_sq);
  *focal_length2 = sqrt(focal_length2_sq);
  return true;
}

}  // namespace theia
