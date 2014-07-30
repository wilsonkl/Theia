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

#include "gtest/gtest.h"

namespace theia {

using Eigen::Matrix3d;
using Eigen::Vector3d;

TEST(FundamentalMatrixUtil, FocalLengths) {
  // Caked example from Matlab
  const double kFundamentalMatrix[3 * 3] = {
    -0.6265377489527094,  -0.1392530208531561,  -1.0420378420644032,
    -0.5139391647114737,   0.9541696721680205,  -0.1957198670050392,
    -0.0250214759315877,   0.5815828325966529,   0.3205687368389822,
  };

  double focal_length1, focal_length2;
  EXPECT_TRUE(FocalLengthsFromFundamentalMatrix(
      kFundamentalMatrix, &focal_length1, &focal_length2));
  EXPECT_NEAR(focal_length1, 1.0 / 1.351, 3e-15);
  EXPECT_NEAR(focal_length2, 1.0 / 0.971, 3e-15);
}

}  // namespace theia
