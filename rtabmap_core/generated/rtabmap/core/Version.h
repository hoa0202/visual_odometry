#ifndef VERSION_H_
#define VERSION_H_

#define RTABMAP_VERSION "0.21.4"
#define RTABMAP_VERSION_MAJOR 0
#define RTABMAP_VERSION_MINOR 21
#define RTABMAP_VERSION_PATCH 4
#define RTABMAP_VERSION_COMPARE(major, minor, patch) \
  (major>=0 || (major==0 && minor>=21) || (major==0 && minor==21 && patch>=4))

#define RTABMAP_GTSAM
#define RTABMAP_VERTIGO
#define RTABMAP_ZED

#include <pcl/pcl_config.h>
#if PCL_VERSION_COMPARE(>, 1, 11, 1)
#include <pcl/types.h>
#define RTABMAP_PCL_INDEX pcl::index_t
#elif PCL_VERSION_COMPARE(>=, 1, 10, 0)
#define RTABMAP_PCL_INDEX std::uint32_t
#else
#include <pcl/pcl_macros.h>
#define RTABMAP_PCL_INDEX pcl::uint32_t
#endif

#endif
