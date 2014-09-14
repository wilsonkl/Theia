# CMakeFiles.txt for CImg
#
# Haz-Edine Assemlal <assemlal@gmail.com>

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CIMG_PACKAGE_VERSION_MAJOR 1)
set(CIMG_PACKAGE_VERSION_MINOR 5)
set(CIMG_PACKAGE_VERSION_PATCH 0)

# register local modules
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_SOURCE_DIR}/cmake-modules")


PROJECT(cimg)

# Packaging defines
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_VERSION_MAJOR ${CIMG_PACKAGE_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${CIMG_PACKAGE_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${CIMG_PACKAGE_VERSION_PATCH})
include(CPack)

set(PACKAGE "CImg")
set(PACKAGE_BUGREPORT "david.tschumperle@greyc.ensicaen.fr")
set(PACKAGE_NAME "CImg")
set(PACKAGE_VERSION "${CIMG_PACKAGE_VERSION_MAJOR}.${CIMG_PACKAGE_VERSION_MINOR}.${CIMG_PACKAGE_VERSION_PATCH}")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")


option(BUILD_GIMP	  "Build GIMP plug-ins" OFF)
option(BUILD_GMIC	  "Build GMIC" ON)
option(WITH_VT100	  "Enable color output messages " ON)
option(WITH_CIMG_DISPLAY  "Use CImg display capabilities" OFF)
option(WITH_OPENMP	  "Use OpenMP parallel core instructions" ON)
option(WITH_XRANDR	  "Use XRandR library" OFF)
option(WITH_PNG		  "Use PNG library" ON)
option(WITH_JPEG	  "Use JPEG library" ON)
option(WITH_TIFF	  "Use TIFF library" ON)
option(WITH_ZLIB	  "Use Zlib library" ON)
option(WITH_MAGICK	  "Use Magick library" OFF)
option(WITH_LAPACK	  "Use LAPACK library" ON)

if(WITH_VT100)
  add_definitions(-Dcimg_use_vt100)
endif(WITH_VT100)

if( WITH_CIMG_DISPLAY )
  if(UNIX OR APPLE)
    if( CMAKE_SYSTEM_NAME MATCHES "(Solaris|SunOS)" )
       list( APPEND DEPENDENCIES_LIBRARIES "-R$(X11PATH)/lib -lrt -lnsl -lsocket" )
     else()
       find_package( X11 REQUIRED )
       include_directories( ${X11_INCLUDE_DIR} )
       list( APPEND DEPENDENCIES_LIBRARIES ${X11_LIBRARIES} )
     endif()
  elseif(WIN32)
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mwindows" )
    list( APPEND DEPENDENCIES_LIBRARIES "-lgdi32" )
  endif()
else(WITH_CIMG_DISPLAY)
  add_definitions(-Dcimg_display=0)
endif(WITH_CIMG_DISPLAY)

if(WITH_OPENMP)
  find_package( OpenMP QUIET)
  IF (OpenMP_FOUND)
    add_definitions(-Dcimg_use_openmp)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ELSE (OpenMP_FOUND)
    MESSAGE("-- Failed to find OpenMP, disabling.")
  ENDIF(OpenMP_FOUND)
endif(WITH_OPENMP)

if(WITH_XRANDR)
  if(NOT X11_Xrandr_INCLUDE_PATH)
    message(FATAL_ERROR "Failed to find X11 XrandR which is required")
  endif(NOT X11_Xrandr_INCLUDE_PATH)
endif(WITH_XRANDR)

if(WITH_PNG)
  find_package( PNG REQUIRED )
  add_definitions(-Dcimg_use_png ${PNG_DEFINITIONS})
  include_directories( ${PNG_INCLUDE_DIRS} )
  list( APPEND DEPENDENCIES_LIBRARIES ${PNG_LIBRARIES} )
endif(WITH_PNG)

if(WITH_JPEG)
  find_package( JPEG REQUIRED )
  add_definitions( -Dcimg_use_jpeg )
  include_directories( ${JPEG_INCLUDE_DIR} )
  list( APPEND DEPENDENCIES_LIBRARIES ${JPEG_LIBRARIES} )
endif(WITH_JPEG)

if(WITH_TIFF)
  find_package( TIFF REQUIRED )
  add_definitions( -Dcimg_use_tiff )
  include_directories( ${TIFF_INCLUDE_DIR} )
  list( APPEND DEPENDENCIES_LIBRARIES ${TIFF_LIBRARIES} )
endif(WITH_TIFF)

if(WITH_ZLIB)
  find_package( ZLIB REQUIRED )
  add_definitions( -Dcimg_use_zlib )
  include_directories( ${ZLIB_INCLUDE_DIR} )
  list( APPEND DEPENDENCIES_LIBRARIES ${ZLIB_LIBRARIES} )
endif(WITH_ZLIB)

if(WITH_MAGICK)
  find_package( ImageMagick QUIET COMPONENTS Magick++ convert)
  IF (ImageMagick_FOUND)
    add_definitions( -Dcimg_use_magick )
    include_directories( ${ImageMagick_INCLUDE_DIRS} )
    list( APPEND DEPENDENCIES_LIBRARIES ${ImageMagick_LIBRARIES} )
  ELSE (ImageMagick_FOUND)
    MESSAGE("-- Failed to find ImageMagick, disabling.")
  ENDIF(ImageMagick_FOUND)
endif(WITH_MAGICK)

if(WITH_LAPACK)
  find_package( LAPACK QUIET )
  IF (LAPACK_FOUND)
    add_definitions( -Dcimg_use_lapack )
    include_directories( ${LAPACK_INCLUDE_DIRS} )
    list( APPEND DEPENDENCIES_LIBRARIES ${LAPACK_LIBRARIES} )
  ELSE (LAPACK_FOUND)
    MESSAGE("-- Failed to find Lapack, disabling.")
  ENDIF (LAPACK_FOUND)
endif(WITH_LAPACK)

if(WITH_BOARD)
  find_package( BOARD  )
  IF (BOARD_FOUND)
    add_definitions( -Dcimg_use_board )
    include_directories( ${BOARD_INCLUDE_DIR} )
    list( APPEND DEPENDENCIES_LIBRARIES ${BOARD_LIBRARIES} )
  ELSE (BOARD_FOUND)
    MESSAGE("-- Failed to find Board library, disabling.")
  ENDIF (BOARD_FOUND)
endif(WITH_BOARD)

# aliases
set(VERSION "${PACKAGE_VERSION}")
include_directories(  ${CMAKE_CURRENT_SOURCE_DIR} )
ADD_LIBRARY(cimg cimg.cc)
TARGET_LINK_LIBRARIES(cimg ${DEPENDENCIES_LIBRARIES} )

INSTALL(TARGETS cimg
  EXPORT  TheiaExport
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LIB_SUFFIX}
  ARCHIVE DESTINATION lib${LIB_SUFFIX})
