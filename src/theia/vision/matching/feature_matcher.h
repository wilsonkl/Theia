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

#ifndef THEIA_VISION_MATCHING_FEATURE_MATCHER_H_
#define THEIA_VISION_MATCHING_FEATURE_MATCHER_H_

#include <unordered_map>
#include <vector>

#include "theia/util/map_util.h"
#include "theia/util/util.h"
#include "theia/vision/matching/feature_matcher_options.h"

namespace theia {

struct FeatureMatch {
  FeatureMatch() {}
  FeatureMatch(int f1_ind, int f2_ind, float dist)
      : feature1_ind(f1_ind), feature2_ind(f2_ind), distance(dist) {}
  // Index of the feature in the first image.
  int feature1_ind;
  // Index of the feature in the second image.
  int feature2_ind;
  // Distance between the two features.
  float distance;
};

template <class DistanceMetric> class FeatureMatcher {
 public:
  typedef typename DistanceMetric::DistanceType DistanceType;
  typedef typename DistanceMetric::DescriptorType DescriptorType;

  FeatureMatcher() {}
  ~FeatureMatcher() {}

  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  virtual bool Match(
      const FeatureMatcherOptions& options,
      const std::vector<DescriptorType>& desc_1,
      const std::vector<DescriptorType>& desc_2,
      std::vector<FeatureMatch>* matches);

 protected:
  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  virtual void MatchNearestNeighbor(
      const std::vector<DescriptorType>& desc_1,
      const std::vector<DescriptorType>& desc_2,
      std::vector<FeatureMatch>* matches) = 0;

  // The k-nearest neighbors are found for each descriptor in desc_1. Each entry
  // of the output matches is a vector with the k-nearest-neighbors of the
  // descriptor in desc_1. It is assumed that the matches are in ascending
  // order, that is, the best match is first.
  virtual void MatchKNearestNeighbors(
      const int knn,
      const std::vector<DescriptorType>& desc_1,
      const std::vector<DescriptorType>& desc_2,
      std::vector<std::vector<FeatureMatch> >* matches) = 0;

  // Match features from feature set 1 to 2 and keep only the matches that pass
  // Lowe's ratio test where the best NN match distance is less than
  // lowes_ratio * the distance of the second NN match.
  void MatchLowesRatio(const FeatureMatcherOptions& options,
                       const std::vector<DescriptorType>& desc_1,
                       const std::vector<DescriptorType>& desc_2,
                       std::vector<FeatureMatch>* matches);

  void RemoveMatchesAboveMaxMatchDistance(
      const float max_allowed_match_distance,
      std::vector<FeatureMatch>* matches);

  void IntersectMatches(const std::vector<FeatureMatch>& backwards_matches,
                        std::vector<FeatureMatch>* forward_matches);

 private:
  DISALLOW_COPY_AND_ASSIGN(FeatureMatcher);
};

// ---------------------- Implementation ------------------------ //

template <class DistanceMetric>
bool FeatureMatcher<DistanceMetric>::Match(
    const FeatureMatcherOptions& options,
    const std::vector<DescriptorType>& desc_1,
    const std::vector<DescriptorType>& desc_2,
    std::vector<FeatureMatch>* matches) {
  // Find matches between feature set 1 and set 2.
  if (options.use_lowes_ratio) {
    MatchLowesRatio(options, desc_1, desc_2, matches);
  } else {
    MatchNearestNeighbor(desc_1, desc_2, matches);
  }
  RemoveMatchesAboveMaxMatchDistance(options.max_match_distance, matches);

  if (!options.keep_only_symmetric_matches) {
    return true;
  }

  // If we want to keep symmetric matches, then we need to compute the matches
  // from feature set 2 to feature set 1 and get the intersection.
  std::vector<FeatureMatch> backwards_matches;
  if (options.use_lowes_ratio) {
    MatchLowesRatio(options, desc_2, desc_1, &backwards_matches);
  } else {
    MatchNearestNeighbor(desc_2, desc_1, &backwards_matches);
  }
  RemoveMatchesAboveMaxMatchDistance(options.max_match_distance,
                                     &backwards_matches);

  IntersectMatches(backwards_matches, matches);

  return true;
}

template <class DistanceMetric>
void FeatureMatcher<DistanceMetric>::MatchLowesRatio(
    const FeatureMatcherOptions& options,
    const std::vector<DescriptorType>& desc_1,
    const std::vector<DescriptorType>& desc_2,
    std::vector<FeatureMatch>* matches) {
  CHECK_NOTNULL(matches)->reserve(desc_1.size());

  std::vector<std::vector<FeatureMatch> > knn_matches;
  MatchKNearestNeighbors(2, desc_1, desc_2, &knn_matches);
  for (int i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance / knn_matches[i][1].distance <
        options.lowes_ratio) {
      matches->push_back(knn_matches[i][0]);
    }
  }
}

template <class DistanceMetric>
void FeatureMatcher<DistanceMetric>::RemoveMatchesAboveMaxMatchDistance(
    const float max_allowed_match_distance,
    std::vector<FeatureMatch>* matches) {
  auto match_iterator = matches->begin();
  while (match_iterator != matches->end()) {
    if (match_iterator->distance >= max_allowed_match_distance) {
      match_iterator = matches->erase(match_iterator);
    } else {
      ++match_iterator;
    }
  }
}

template <class DistanceMetric>
void FeatureMatcher<DistanceMetric>::IntersectMatches(
    const std::vector<FeatureMatch>& backwards_matches,
    std::vector<FeatureMatch>* forward_matches) {
  std::unordered_map<int, int> index_map;
  index_map.reserve(backwards_matches.size());
  // Add all feature2 -> feature1 matches to the map.
  for (const FeatureMatch& feature_match : backwards_matches) {
    InsertOrDie(&index_map,
                feature_match.feature1_ind,
                feature_match.feature2_ind);
  }

  // Search the map for feature1 -> feature2 matches that are also present in
  // the feature2 -> feature1 matches.
  auto match_iterator = forward_matches->begin();
  while (match_iterator != forward_matches->end()) {
    if (match_iterator->feature1_ind !=
        FindWithDefault(index_map, match_iterator->feature2_ind, -1)) {
      match_iterator = forward_matches->erase(match_iterator);
      continue;
    }

    ++match_iterator;
  }
}

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_FEATURE_MATCHER_H_
