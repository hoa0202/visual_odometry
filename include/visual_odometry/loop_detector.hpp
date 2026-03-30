#pragma once

#include "visual_odometry/types.hpp"
#include "visual_odometry/bow_database.hpp"
#include "visual_odometry/bayesian_filter.hpp"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>

namespace vo {

class LoopDetector {
public:
    struct LoopResult {
        bool detected{false};
        int query_kf_id{-1};
        int match_kf_id{-1};
        cv::Mat T_query_from_match;  // 4x4 relative pose (optical, mm)
        double score{0.0};
    };

    LoopDetector(const std::string& vocab_path,
                 double bayesian_threshold = 0.55,
                 int temporal_consistency = 3,
                 int min_matches = 30,
                 double min_score = 0.3);

    bool isReady() const;

    void addSignature(int sig_id, const cv::Mat& descriptors,
                      const std::vector<cv::KeyPoint>& keypoints);

    void removeSignature(int sig_id);

    // Set neighbor relationships for Bayesian transition model
    void setNeighbors(int sig_id, const std::vector<int>& neighbor_ids);

    // Detect loop closure using Bayesian filter over BoW similarities
    LoopResult detect(int query_sig_id,
                      const cv::Mat& query_descriptors,
                      const std::vector<cv::KeyPoint>& query_keypoints,
                      int min_kf_gap,
                      const CameraParams& cam);

    // BoW similarity between two signatures (for rehearsal)
    double similarity(int sig_a, int sig_b) const;

    void reset();

private:
    std::unique_ptr<BowDatabase> bow_db_;
    BayesianFilter bayes_filter_;
    int min_matches_;
    double min_score_;

    struct SigData {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints;
    };
    std::map<int, SigData> sig_data_;

    bool geometricVerification(
        const std::vector<cv::Point2f>& pts_q,
        const std::vector<cv::Point2f>& pts_m,
        const CameraParams& cam,
        cv::Mat& T_out);

    int matchDescriptors(const cv::Mat& desc_a, const cv::Mat& desc_b,
                         std::vector<cv::DMatch>& good_matches);
};

}  // namespace vo
