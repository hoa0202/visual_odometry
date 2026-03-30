#include "visual_odometry/loop_detector.hpp"
#include <opencv2/calib3d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>

namespace vo {

static rclcpp::Logger llog() { return rclcpp::get_logger("loop_detector"); }
static auto lclock() {
    static auto c = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
    return c;
}

LoopDetector::LoopDetector(const std::string& vocab_path,
                           double bayesian_threshold,
                           int temporal_consistency,
                           int min_matches,
                           double min_score)
    : bow_db_(std::make_unique<BowDatabase>(vocab_path))
    , bayes_filter_(bayesian_threshold, temporal_consistency)
    , min_matches_(min_matches)
    , min_score_(min_score)
{}

bool LoopDetector::isReady() const { return bow_db_ && bow_db_->isReady(); }

void LoopDetector::addSignature(int sig_id, const cv::Mat& descriptors,
                                const std::vector<cv::KeyPoint>& keypoints) {
    if (!isReady() || descriptors.empty()) return;

    bow_db_->addSignature(sig_id, descriptors);
    bayes_filter_.addLocation(sig_id);

    SigData sd;
    sd.descriptors = descriptors.clone();
    sd.keypoints = keypoints;
    sig_data_[sig_id] = sd;
}

void LoopDetector::removeSignature(int sig_id) {
    bow_db_->removeSignature(sig_id);
    bayes_filter_.removeLocation(sig_id);
    sig_data_.erase(sig_id);
}

void LoopDetector::setNeighbors(int sig_id, const std::vector<int>& neighbor_ids) {
    bayes_filter_.setNeighbors(sig_id, neighbor_ids);
}

double LoopDetector::similarity(int sig_a, int sig_b) const {
    if (!isReady()) return 0.0;
    return bow_db_->score(sig_a, sig_b);
}

LoopDetector::LoopResult LoopDetector::detect(
    int query_sig_id,
    const cv::Mat& query_descriptors,
    const std::vector<cv::KeyPoint>& query_keypoints,
    int min_kf_gap,
    const CameraParams& cam)
{
    LoopResult result;
    result.query_kf_id = query_sig_id;

    if (!isReady() || query_descriptors.empty() || sig_data_.size() < 3)
        return result;

    // 1. Compute BoW likelihoods against all WM signatures
    auto likelihoods = bow_db_->computeLikelihoods(query_sig_id);

    // Filter out too-recent signatures (within min_kf_gap)
    for (auto it = likelihoods.begin(); it != likelihoods.end(); ) {
        if (std::abs(it->first - query_sig_id) < min_kf_gap)
            it = likelihoods.erase(it);
        else
            ++it;
    }

    if (likelihoods.empty()) return result;

    // 2. Bayesian filter update
    auto hypothesis = bayes_filter_.update(likelihoods, query_sig_id);

    if (hypothesis.sig_id < 0) return result;

    // 3. Geometric verification on the Bayesian winner
    auto it_match = sig_data_.find(hypothesis.sig_id);
    if (it_match == sig_data_.end()) return result;

    // ORB descriptor matching for point correspondences
    std::vector<cv::DMatch> good_matches;
    int num_good = matchDescriptors(query_descriptors, it_match->second.descriptors,
                                     good_matches);

    if (num_good < min_matches_) {
        RCLCPP_DEBUG(llog(), "Loop candidate KF%d rejected: %d matches < %d",
            hypothesis.sig_id, num_good, min_matches_);
        return result;
    }

    // Extract matched point pairs
    std::vector<cv::Point2f> pts_q, pts_m;
    for (const auto& m : good_matches) {
        if (m.queryIdx < static_cast<int>(query_keypoints.size()) &&
            m.trainIdx < static_cast<int>(it_match->second.keypoints.size())) {
            pts_q.push_back(query_keypoints[m.queryIdx].pt);
            pts_m.push_back(it_match->second.keypoints[m.trainIdx].pt);
        }
    }

    cv::Mat T_rel;
    if (pts_q.size() >= 8 && geometricVerification(pts_q, pts_m, cam, T_rel)) {
        result.detected = true;
        result.match_kf_id = hypothesis.sig_id;
        result.T_query_from_match = T_rel;
        result.score = hypothesis.probability;

        RCLCPP_INFO(llog(), "Loop closure (Bayesian): KF%d↔KF%d P=%.3f matches=%d",
            query_sig_id, hypothesis.sig_id, hypothesis.probability, num_good);
    }

    return result;
}

int LoopDetector::matchDescriptors(const cv::Mat& desc_a, const cv::Mat& desc_b,
                                    std::vector<cv::DMatch>& good_matches) {
    good_matches.clear();
    if (desc_a.empty() || desc_b.empty()) return 0;

    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher->knnMatch(desc_a, desc_b, knn, 2);

    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < 0.7f * pair[1].distance)
            good_matches.push_back(pair[0]);
    }
    return static_cast<int>(good_matches.size());
}

bool LoopDetector::geometricVerification(
    const std::vector<cv::Point2f>& pts_q,
    const std::vector<cv::Point2f>& pts_m,
    const CameraParams& cam,
    cv::Mat& T_out)
{
    cv::Mat K = cam.getCameraMatrix();
    cv::Mat E = cv::findEssentialMat(pts_q, pts_m, K, cv::RANSAC, 0.999, 1.0);
    if (E.empty()) return false;

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts_q, pts_m, K, R, t);
    double inlier_ratio = static_cast<double>(inliers) / pts_q.size();
    if (inliers < 12 || inlier_ratio < 0.4) return false;

    T_out = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T_out(cv::Rect(0,0,3,3)));
    t.copyTo(T_out(cv::Rect(3,0,1,3)));
    return true;
}

void LoopDetector::reset() {
    sig_data_.clear();
    bayes_filter_.reset();
    bow_db_ = std::make_unique<BowDatabase>("");
}

}  // namespace vo
