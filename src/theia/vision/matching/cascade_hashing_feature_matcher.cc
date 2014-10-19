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

#include "theia/vision/matching/cascade_hashing_feature_matcher.h"

#include <Eigen/Core>
#include <glog/logging.h>

#include <algorithm>

#include <type_traits>
#include <vector>

#include "theia/vision/matching/cascade_hasher.h"
#include "theia/vision/matching/feature_matcher.h"
#include "theia/vision/matching/feature_matcher_utils.h"

namespace theia {

bool CascadeHashingFeatureMatcher::Match(
    const FeatureMatcherOptions& options,
    const std::vector<Eigen::VectorXf>& desc_1,
    const std::vector<Eigen::VectorXf>& desc_2,
    std::vector<FeatureMatch>* matches) {
  typedef Eigen::Matrix<int, 128, 1> Vector128i;

  CHECK_NOTNULL(matches)->clear();
  matches->reserve(desc_1.size());

  CascadeHasher hasher;
  CHECK(hasher.Initialize());

  HashedImage hashed_desc_1, hashed_desc_2;
  hasher.CreateHashedSiftDescriptors(desc_1, &hashed_desc_1);
  hasher.CreateHashedSiftDescriptors(desc_2, &hashed_desc_2);

  hasher.MatchImages(hashed_desc_1,
                     hashed_desc_2,
                     options.lowes_ratio,
                     matches);

  // Perform symmetric matching if appropriate.
  if (options.keep_only_symmetric_matches) {
    std::vector<FeatureMatch> backwards_matches;
    hasher.MatchImages(hashed_desc_2,
                       hashed_desc_1,
                       options.lowes_ratio,
                       &backwards_matches);
    IntersectMatches(backwards_matches, matches);
  }

  return true;
}

}  // namespace theia
