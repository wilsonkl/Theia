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
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <Eigen/SVD>
#include <glog/logging.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "theia/util/map_util.h"
#include "theia/vision/sfm/pose/util.h"

namespace theia {

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;

typedef Eigen::SparseMatrix<double> SparseMatrix;

namespace {

// Set up linear system R_j - R_{i,j} * R_i = 0. This is a (3 * m) x (3 * n)
// system where m is the number of relative rotations and n is the number of
// views. Sets up the linear system lhs * x = rhs.
void SetupSparseLinearSystem(
    const std::vector<PairwiseRelativeRotation>& relative_rotations,
    const std::unordered_map<int, int>& view_id_map,
    SparseMatrix* lhs,
    MatrixXd* rhs) {
  typedef Eigen::Triplet<double> TripletEntry;

  static const int kRotationMatrixDimSize = 3;

  // Reserve size of matrices.
  *lhs = SparseMatrix(kRotationMatrixDimSize * relative_rotations.size() +
                          kRotationMatrixDimSize,
                      kRotationMatrixDimSize * view_id_map.size());
  *rhs = MatrixXd::Zero(kRotationMatrixDimSize * relative_rotations.size() +
                            kRotationMatrixDimSize,
                        kRotationMatrixDimSize);

  // Iterate through the relative rotation constraints and add them to the
  // sparse matrix.
  std::vector<TripletEntry> triplet_list;
  triplet_list.reserve(12 * relative_rotations.size());
  for (int i = 0; i < relative_rotations.size(); i++) {
    const int view1_index =
        FindOrDie(view_id_map, relative_rotations[i].view_id_one);
    const int view2_index =
        FindOrDie(view_id_map, relative_rotations[i].view_id_two);

    // Set R_j term to identity.
    for (int j = 0; j < 3; j++) {
      triplet_list.push_back(
          TripletEntry(kRotationMatrixDimSize * i + j,
                       kRotationMatrixDimSize * view2_index + j,
                       relative_rotations[i].weight));
    }

    // Set R_i term. This is equal to the weight times the relative rotation
    // (note the negative sign).
    const Matrix3d relative_rotation =
        -relative_rotations[i].weight * relative_rotations[i].rotation;
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        triplet_list.push_back(TripletEntry(
            kRotationMatrixDimSize * i + r,
            kRotationMatrixDimSize * view1_index + c,
            relative_rotation(r, c)));
      }
    }
  }

  // Set up the rhs matrix (i.e. b in Ax = b). All entries will be zero except
  // for one constraint that keeps the first rotation equal to the identity so
  // that the solution to the system is not trivial. Without this constraint, we
  // will have to solve for the null space which makes solving the system much
  // more complicated (and, indeed, solving the linear system would not be the
  // correct method).
  for (int i = 0; i < 3; i++) {
    triplet_list.push_back(TripletEntry(
        kRotationMatrixDimSize * relative_rotations.size() + i, i, 1.0));
  }

  rhs->block<kRotationMatrixDimSize, kRotationMatrixDimSize>(
      kRotationMatrixDimSize * relative_rotations.size(), 0) =
      Matrix3d::Identity();

  lhs->setFromTriplets(triplet_list.begin(), triplet_list.end());
}

}  // namespace

bool GlobalOrientationLinear(
    const std::vector<PairwiseRelativeRotation>& relative_rotations,
    std::vector<GlobalOrientation>* global_orientations) {
  CHECK_GT(relative_rotations.size(), 0);
  CHECK_NOTNULL(global_orientations);

  // Determine all unique view ids.
  std::vector<int> view_ids;
  for (const auto& view_pair : relative_rotations) {
    view_ids.push_back(view_pair.view_id_one);
    view_ids.push_back(view_pair.view_id_two);
  }

  // Sort and unique the vector.
  std::sort(view_ids.begin(), view_ids.end());
  view_ids.erase(std::unique(view_ids.begin(), view_ids.end()), view_ids.end());

  // Lookup map to keep track of the global orientation estimates by view id.
  std::unordered_map<int, int> view_id_map;

  // Initialize all global orientation ids and populate view id map.
  global_orientations->resize(view_ids.size());
  for (int i = 0; i < global_orientations->size(); i++) {
    global_orientations->at(i).view_id = view_ids[i];
    view_id_map[view_ids[i]] = i;
  }

  // Setup the sparse linear system.
  SparseMatrix lhs;
  MatrixXd rhs;
  SetupSparseLinearSystem(relative_rotations, view_id_map, &lhs, &rhs);

  // Setup sparse QR solver.
  // TODO(cmsweeney): Use SuiteSparse instead.
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int> >
      sparse_qr_solver(lhs);

  // Check that the input to SparseQR was valid and could be succesffully
  // factorized.
  if (sparse_qr_solver.info() != Eigen::Success) {
    VLOG(2) << "The sparse matrix of relative rotation constraints could not "
               "be factorized.";
    return false;
  }

  // Solve the linear least squares system.
  const MatrixXd solution = sparse_qr_solver.solve(rhs);

  // Check that the system could be solved successfully.
  if (sparse_qr_solver.info() != Eigen::Success) {
    VLOG(2) << "The sparse linear system could not be solved.";
    return false;
  }

  // Project all solutions into a valid SO3 rotation space. The linear system
  // above makes no constraint on the space of the solutions, so the final
  // solutions are not guaranteed to be valid rotations (e.g., det(R) may not be
  // +1).
  for (int i = 0; i < global_orientations->size(); i++) {
    const Matrix3d& solution_i = solution.block<3, 3>(3 * i, 0);
    global_orientations->at(i).rotation = ProjectToRotationMatrix(solution_i);
  }

  return true;
}

}  // namespace theia
