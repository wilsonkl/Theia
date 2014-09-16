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

#ifndef THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_
#define THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_

#include <algorithm>
#include <vector>

#include "theia/vision/matching/feature_matcher.h"

namespace theia {

// Performs features matching between two sets of features using a brute force
// matching method.
template <class DistanceMetric>
class BruteForceFeatureMatcher : public FeatureMatcher<DistanceMetric> {
 public:
  typedef typename DistanceMetric::DistanceType DistanceType;
  typedef typename DistanceMetric::DescriptorType DescriptorType;

  BruteForceFeatureMatcher() {}
  ~BruteForceFeatureMatcher() {}

 protected:
  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  virtual void MatchNearestNeighbor(
      const std::vector<DescriptorType>& desc_1,
      const std::vector<DescriptorType>& desc_2,
      std::vector<FeatureMatch>* matches);

  // The k-nearest neighbors are found for each descriptor in desc_1. Each entry
  // of the output matches is a vector with the k-nearest-neighbors of the
  // descriptor in desc_1. It is assumed that the matches are in ascending
  // order, that is, the best match is first.
  virtual void MatchKNearestNeighbors(
      const int knn,
      const std::vector<DescriptorType>& desc_1,
      const std::vector<DescriptorType>& desc_2,
      std::vector<std::vector<FeatureMatch> >* matches);

 private:
  bool CompareFeaturesByDistance(const FeatureMatch& lhs,
                                 const FeatureMatch& rhs) {
    return lhs.distance < rhs.distance;
  }

  DISALLOW_COPY_AND_ASSIGN(BruteForceFeatureMatcher);
};

// ---------------------- Implementation ------------------------ //

template <class DistanceMetric>
void BruteForceFeatureMatcher<DistanceMetric>::MatchNearestNeighbor(
    const std::vector<DescriptorType>& desc_1,
    const std::vector<DescriptorType>& desc_2,
    std::vector<FeatureMatch>* matches) {
  CHECK_NOTNULL(matches)->reserve(desc_1.size());

  DistanceMetric distance;
  for (int i = 0; i < desc_1.size(); i++) {
    FeatureMatch best_match(i, 0, distance(desc_1[i], desc_2[0]));
    for (int j = 1; j < desc_2.size(); j++) {
      const DistanceType temp_dist = distance(desc_1[i], desc_2[j]);
      if (temp_dist < best_match.distance) {
        best_match.feature2_ind = j;
        best_match.distance = temp_dist;
      }
    }
    matches->emplace_back(best_match);
  }
}

namespace brute_force_feature_matcher {
inline bool CompareFeaturesByDistance(const FeatureMatch& feature1,
                                      const FeatureMatch& feature2) {
  return feature1.distance < feature2.distance;
}

}  // namespace brute_force_feature_matcher

template <class DistanceMetric>
void BruteForceFeatureMatcher<DistanceMetric>::MatchKNearestNeighbors(
    const int knn,
    const std::vector<DescriptorType>& desc_1,
    const std::vector<DescriptorType>& desc_2,
    std::vector<std::vector<FeatureMatch> >* matches) {
  CHECK_NOTNULL(matches)->resize(desc_1.size());
  DistanceMetric distance;

  for (int i = 0; i < desc_1.size(); i++) {
    std::vector<FeatureMatch> knn_matches(desc_2.size());
    // Get all feature match distances to desc_1[i].
    for (int j = 0; j < desc_2.size(); j++) {
      knn_matches[j] = FeatureMatch(i, j, distance(desc_1[i], desc_2[j]));
    }

    // Perform a partial sort on the elements to get the k min elements. This
    // should be rather efficient.
    std::partial_sort(knn_matches.begin(),
                      knn_matches.begin() + knn,
                      knn_matches.end(),
                      brute_force_feature_matcher::CompareFeaturesByDistance);
    // Resize so we only keep the first k elements.
    knn_matches.resize(knn);

    (*matches)[i] = knn_matches;
  }
}

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_
