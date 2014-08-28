IF (${BUILD_TESTING})
  ADD_SUBDIRECTORY(gtest)
ENDIF (${BUILD_TESTING})

# Add Agast keypoint detector
ADD_SUBDIRECTORY(agast)

# Add Lemon graph library
ADD_SUBDIRECTORY(lemon)

# Add Optimo
ADD_SUBDIRECTORY(optimo)

# Add Statx
ADD_SUBDIRECTORY(statx)

# Add VLFeat
ADD_SUBDIRECTORY(vlfeat)
