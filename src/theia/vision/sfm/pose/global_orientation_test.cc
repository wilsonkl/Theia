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

#include "theia/vision/sfm/pose/global_orientation.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "theia/math/util.h"
#include "theia/util/hash.h"
#include "theia/util/map_util.h"
#include "theia/util/random.h"
#include "theia/vision/sfm/pose/util.h"
#include "theia/vision/sfm/types.h"

namespace theia {

using Eigen::AngleAxisd;
using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {
// Noise to be added to the relative rotations. Noise is applied about each of
// the x, y, and z axes.
static const double kRelativeRotationNoiseDegrees = 1.0;

// Computes the relative rotation between two views given the global
// orientations.
void GetRelativeRotationFromGlobalOrientation(
    const Matrix3d& global_orientation1,
    const Matrix3d& global_orientation2,
    const double weight,
    const double pairwise_rotation_noise_degrees,
    RelativeRotation* relative_rotation) {
  relative_rotation->rotation =
      global_orientation2 * global_orientation1.transpose();

  // Add noise if applicable.
  if (pairwise_rotation_noise_degrees) {
    const Vector3d noise_std_dev =
        pairwise_rotation_noise_degrees * Vector3d::Random();

    // Apply the rotation noise about each axis.
    Matrix3d rotation_noise =
        (AngleAxisd(Radians(noise_std_dev(0)), Vector3d::UnitX()),
         AngleAxisd(Radians(noise_std_dev(1)), Vector3d::UnitY()),
         AngleAxisd(Radians(noise_std_dev(2)), Vector3d::UnitZ()))
            .toRotationMatrix();
    relative_rotation->rotation = rotation_noise * relative_rotation->rotation;
  }

  relative_rotation->weight = weight;
}

// Computes the global orientation estimation from the pairwise relative
// rotations. CHECKs that the estimated orientations are close to the ground
// truth orientations with an angular tolerance.
void TestGlobalOrientationLinear(
    const std::unordered_map<ViewId, Matrix3d>& gt_global_orientation,
    const std::vector<ViewIdPair>& pairwise_indices,
    const std::vector<double>& weights,
    const double relative_rotation_noise,
    const double tolerance_in_degrees) {
  // Compute the relative rotation.
  std::unordered_map<ViewIdPair, RelativeRotation> relative_rotations;
  for (int i = 0; i < pairwise_indices.size(); i++) {
    RelativeRotation relative_rotation;
    GetRelativeRotationFromGlobalOrientation(
        FindOrDie(gt_global_orientation, pairwise_indices[i].first),
        FindOrDie(gt_global_orientation, pairwise_indices[i].second),
        weights[i],
        relative_rotation_noise,
        &relative_rotation);
    const ViewIdPair view_id_pair =
        ViewIdPair(pairwise_indices[i].first, pairwise_indices[i].second);
    relative_rotations[view_id_pair] = relative_rotation;
  }

  // Run global orientation estimation.
  std::unordered_map<ViewId, Matrix3d> estimated_global_orientation;
  CHECK(GlobalOrientationLinear(relative_rotations,
                                &estimated_global_orientation));

  // Check that the estimated and ground truth orientations are the same size.
  for (int i = 0; i < estimated_global_orientation.size(); i++) {
    const Matrix3d loop_matrix =
        FindOrDie(gt_global_orientation, i).transpose() *
        estimated_global_orientation[i];
    const double angular_difference = Degrees(AngleAxisd(loop_matrix).angle());
    CHECK_LE(angular_difference, tolerance_in_degrees);
  }
}

TEST(GlobalOrientationLinear, IdentityRotation) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[i] = Matrix3d::Identity();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 1 },
                                                     { 0, 2 },
                                                     { 1, 2 } };
  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              0.0, 0.0);
}

TEST(GlobalOrientationLinear, SimpleRotationNoNoise) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[i] =
        AngleAxisd(Radians(i * 10.0), Vector3d::UnitY()).toRotationMatrix();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 1 },
                                                     { 0, 2 },
                                                     { 1, 2 } };
  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              0.0, 0.0);
}


TEST(GlobalOrientationLinear, SimpleRotationWithNoise) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[i] =
        AngleAxisd(Radians(i * 10.0), Vector3d::UnitY()).toRotationMatrix();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 1 },
                                                     { 0, 2 },
                                                     { 1, 2 } };
  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              kRelativeRotationNoiseDegrees, 1.0);
}

TEST(GlobalOrientationLinear, NinetyDegreeRotation) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[i] =
        AngleAxisd(Radians(i * 90.0), Vector3d::UnitY()).toRotationMatrix();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 1 },
                                                     { 0, 2 },
                                                     { 1, 2 } };
  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              kRelativeRotationNoiseDegrees, 1.0);
}

TEST(GlobalOrientationLinear, NonConsecutiveIds) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[3 * i] =
        AngleAxisd(Radians(i * 10.0), Vector3d::UnitY()).toRotationMatrix();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 3 },
                                                     { 0, 6 },
                                                     { 3, 6 } };
  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              kRelativeRotationNoiseDegrees, 1.0);
}

TEST(GlobalOrientationLinear, NonuniformWeighting) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;

  for (int i = 0; i < 3; i++) {
    gt_global_orientation[i] =
        AngleAxisd(Radians(i * 10.0), Vector3d::UnitY()).toRotationMatrix();
  }

  const std::vector<ViewIdPair> pairwise_indices = { { 0, 1 },
                                                     { 0, 2 },
                                                     { 1, 2 } };
  const std::vector<double> weights = { 0.2, 1.5, 1.3 };
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              kRelativeRotationNoiseDegrees, 1.0);
}

TEST(GlobalOrientationLinear, ManyViews) {
  std::unordered_map<ViewId, Matrix3d> gt_global_orientation;
  static const int kNumViews = 100;

  InitRandomGenerator();

  gt_global_orientation[0] = Matrix3d::Identity();

  for (int i = 1; i < kNumViews; i++) {
    // Generate a random rotation within 60 deg around a random axis
    gt_global_orientation[i] =
        AngleAxisd(Radians(RandDouble(0, 60.0)),
                   Vector3d::Random().normalized()).toRotationMatrix();
  }

  static const int kNumRelativeRotations = 150;
  std::vector<ViewIdPair> pairwise_indices;

  // Add a relative rotation containing each view so that there is guaranteed to
  // be a valid solution.
  for (int i = 0; i < kNumViews - 1; i++) {
    pairwise_indices.push_back(std::make_pair(i, i + 1));
  }

  // Now add extra constraints with random view pairs.
  for (int i = kNumViews; i < kNumRelativeRotations; i++) {
    const int j = RandInt(0, kNumViews - 2);
    const int k = RandInt(j + 1, kNumViews - 1);
    pairwise_indices.push_back(std::make_pair(j, k));
  }

  const std::vector<double> weights(pairwise_indices.size(), 1.0);
  TestGlobalOrientationLinear(gt_global_orientation, pairwise_indices, weights,
                              kRelativeRotationNoiseDegrees, 5.0);
}

}  // namespace

}  // namespace theia
