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

#include <vector>

#include "gtest/gtest.h"
#include "theia/vision/matching/feature_matcher.h"
#include "theia/vision/matching/feature_matcher_utils.h"

namespace theia {

TEST(FeatureMatcherUtils, FilterByLowesRatio) {
  FeatureMatcherOptions options;
  std::vector<std::vector<FeatureMatch> > feature_matches = {
    { FeatureMatch(0, 1, 0.79), FeatureMatch(0, 2, 1.0) },
    { FeatureMatch(1, 2, 0.9), FeatureMatch(1, 3, 1.0) }
  };

  std::vector<FeatureMatch> matches;
  FilterByLowesRatio(options, feature_matches, &matches);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].feature1_ind, feature_matches[0][0].feature1_ind);
  EXPECT_EQ(matches[0].feature2_ind, feature_matches[0][0].feature2_ind);
}

TEST(FeatureMatcherUtils, RemoveMatchesAboveMaxMatchDistance) {
  std::vector<FeatureMatch> feature_matches = { FeatureMatch(0, 1, 0.8),
                                                FeatureMatch(0, 2, 1.0),
                                                FeatureMatch(1, 2, 0.3),
                                                FeatureMatch(1, 3, 0.9) };
  RemoveMatchesAboveMaxMatchDistance(0.85, &feature_matches);
  EXPECT_EQ(feature_matches.size(), 2);
}


TEST(FeatureMatcherUtils, IntersectMatches) {
  FeatureMatcherOptions options;
  std::vector<FeatureMatch> matches = { FeatureMatch(0, 1, 0.8),
                                         FeatureMatch(1, 2, 1.0) };
  std::vector<FeatureMatch> reverse_matches = { FeatureMatch(1, 0, 0.8),
                                                FeatureMatch(2, 3, 1.0) };
  IntersectMatches(reverse_matches, &matches);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].feature1_ind, 0);
  EXPECT_EQ(matches[0].feature2_ind, 1);
}

}  // namespace theia
