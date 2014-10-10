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

#ifndef THEIA_VISION_MATCHING_FEATURE_MATCHER_UTILS_H_
#define THEIA_VISION_MATCHING_FEATURE_MATCHER_UTILS_H_

#include <vector>

#include "theia/vision/matching/feature_matcher.h"
#include "theia/vision/matching/feature_matcher_options.h"

namespace theia {

// Match features from feature set 1 to 2 and keep only the matches that pass
// Lowe's ratio test where the best NN match distance is less than lowes_ratio *
// the distance of the second NN match. The knn matches must contain the
// k-nearest neighbors in sorted order (lowest distance first).
void FilterByLowesRatio(
    const FeatureMatcherOptions& options,
    const std::vector<std::vector<FeatureMatch> >& knn_matches,
    std::vector<FeatureMatch>* matches);

// Removes all matches whose distance is greater than the max allowed distance.
void RemoveMatchesAboveMaxMatchDistance(
    const float max_allowed_match_distance,
    std::vector<FeatureMatch>* matches);

// Modifies forward matches so that it removes all matches that are not
// contained in the backwards matches.
void IntersectMatches(const std::vector<FeatureMatch>& backwards_matches,
                      std::vector<FeatureMatch>* forward_matches);

// Used for sorting a vector of the feature matches.
inline bool CompareFeaturesByDistance(const FeatureMatch& feature1,
                                      const FeatureMatch& feature2) {
  return feature1.distance < feature2.distance;
}

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_FEATURE_MATCHER_UTILS_H_
