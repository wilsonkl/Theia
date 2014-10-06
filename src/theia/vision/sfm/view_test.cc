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

#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

TEST(View, Constructors) {
  // Default constructor.
  View view;
  EXPECT_EQ(view.Name().compare(""), 0);
  EXPECT_EQ(view.Id(), kInvalidViewId);
  EXPECT_TRUE(!view.IsEstimated());

  // With a name and id.
  View view2("my_view", 1);
  EXPECT_EQ(view2.Name().compare("my_view"), 0);
  EXPECT_EQ(view2.Id(), 1);
  EXPECT_TRUE(!view2.IsEstimated());

  // Copy constructor.
  View view3(view2);
  EXPECT_EQ(view3.Name().compare("my_view"), 0);
  EXPECT_EQ(view3.Id(), 1);
  // Change view 2 and make sure it does not change view 3.
  view2.SetEstimated(true);
  EXPECT_TRUE(view2.IsEstimated());
  EXPECT_TRUE(!view3.IsEstimated());

  // Assign operator.
  View view4 = view3;
  EXPECT_EQ(view3.Name().compare("my_view"), 0);
  EXPECT_EQ(view3.Id(), 1);
  // Change view 3 and make sure it does not change view 4.
  view3.SetEstimated(true);
  EXPECT_TRUE(view3.IsEstimated());
  EXPECT_TRUE(!view4.IsEstimated());
}

TEST(View, GettersAndSetters) {
  View view("my_view", 0);

  EXPECT_EQ(view.Name().compare("my_view"), 0);

  view.SetEstimated(true);
  EXPECT_TRUE(view.IsEstimated());
}

TEST(View, AddRemoveFeatures) {
  View view;

  EXPECT_EQ(view.NumFeatures(), 0);

  std::vector<Feature> features = { Feature(0, 0, 0),
                                    Feature(1, 100, 100),
                                    Feature(2, 50, 50) };

  for (const Feature& feature : features) {
    view.AddFeature(feature);
    EXPECT_TRUE(view.IsTrackVisible(feature.track_id));

    const Feature* retrieved_feature = view.GetFeature(feature.track_id);
    EXPECT_NE(retrieved_feature, nullptr);
    EXPECT_EQ(feature, *retrieved_feature);
  }

  EXPECT_EQ(view.NumFeatures(), 3);

  for (const Feature& feature : features) {
    EXPECT_TRUE(view.RemoveFeature(feature.track_id));
  }

  EXPECT_EQ(view.NumFeatures(), 0);
  EXPECT_TRUE(!view.RemoveFeature(kInvalidTrackId));
}

}  // namespace theia
