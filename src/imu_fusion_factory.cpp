#include "visual_odometry/imu_fusion.hpp"
#include "visual_odometry/imu_fusion_ekf.hpp"

namespace vo {

std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha) {
    return createImuFusion(mode, alpha, EKFParams{});
}

std::unique_ptr<ImuFusionBase> createImuFusion(const std::string& mode, double alpha,
                                               const EKFParams& ekf_params) {
    if (mode == "complementary") {
        return std::make_unique<ComplementaryFilter>(alpha);
    }
    if (mode == "ekf") {
        return std::make_unique<ImuFusionEKF>(ekf_params);
    }
    if (mode == "factor_graph") {
        return std::make_unique<ImuFusionFactorGraph>();
    }
    return nullptr;
}

}  // namespace vo
