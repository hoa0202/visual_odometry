# Visual Odometry State Message
std_msgs/Header header

# Camera pose
geometry_msgs/Pose pose

# Feature detection info
uint32 num_features
uint32 num_matches

# Quality metrics
float32 tracking_quality  # 0.0 to 1.0
float32 scale_confidence  # Scale estimation confidence

# Processing info
float32 processing_time  # milliseconds 