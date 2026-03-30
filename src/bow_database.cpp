#include "visual_odometry/bow_database.hpp"
#include <DBoW2.h>
#include <rclcpp/rclcpp.hpp>
#include <fstream>

namespace vo {

static rclcpp::Logger blog() { return rclcpp::get_logger("bow_database"); }

struct BowDatabase::Impl {
    std::shared_ptr<OrbVocabulary> vocab;
    std::map<int, DBoW2::BowVector> bow_vectors;    // sig_id → BoW
    std::map<int, DBoW2::FeatureVector> feat_vectors; // sig_id → feature vec
};

BowDatabase::BowDatabase(const std::string& vocab_path)
    : impl_(std::make_unique<Impl>())
{
    impl_->vocab = std::make_shared<OrbVocabulary>();

    std::string path = vocab_path;
    if (path.empty()) {
        std::vector<std::string> candidates = {
            "install/visual_odometry/share/visual_odometry/config/vocabulary/ORBvoc.txt",
            "src/visual_odometry/config/vocabulary/ORBvoc.txt",
            "config/vocabulary/ORBvoc.txt",
            "../config/vocabulary/ORBvoc.txt",
        };
        for (const auto& c : candidates) {
            std::ifstream f(c);
            if (f.good()) { path = c; break; }
        }
    }

    if (path.empty()) {
        RCLCPP_ERROR(blog(), "No ORB vocabulary file found — loop closure disabled");
        return;
    }

    RCLCPP_INFO(blog(), "Loading ORB vocabulary: %s ...", path.c_str());
    try {
        impl_->vocab->load(path);
        vocab_loaded_ = true;
        RCLCPP_INFO(blog(), "ORB vocabulary loaded: %u words, %d levels",
            impl_->vocab->size(), impl_->vocab->getDepthLevels());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(blog(), "Failed to load vocabulary: %s", e.what());
    }
}

BowDatabase::~BowDatabase() = default;

int BowDatabase::addSignature(int sig_id, const cv::Mat& descriptors) {
    if (!vocab_loaded_ || descriptors.empty()) return -1;

    // Convert cv::Mat rows to vector<cv::Mat> (DBoW2 format)
    std::vector<cv::Mat> desc_vec;
    desc_vec.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i)
        desc_vec.push_back(descriptors.row(i));

    DBoW2::BowVector bv;
    DBoW2::FeatureVector fv;
    impl_->vocab->transform(desc_vec, bv, fv, impl_->vocab->getDepthLevels() - 2);

    impl_->bow_vectors[sig_id] = bv;
    impl_->feat_vectors[sig_id] = fv;
    return sig_id;
}

double BowDatabase::score(int a, int b) const {
    if (!vocab_loaded_) return 0.0;
    auto it_a = impl_->bow_vectors.find(a);
    auto it_b = impl_->bow_vectors.find(b);
    if (it_a == impl_->bow_vectors.end() || it_b == impl_->bow_vectors.end())
        return 0.0;
    return impl_->vocab->score(it_a->second, it_b->second);
}

std::map<int, double> BowDatabase::query(int query_sig_id, int top_k) const {
    std::map<int, double> results;
    if (!vocab_loaded_) return results;

    auto it_q = impl_->bow_vectors.find(query_sig_id);
    if (it_q == impl_->bow_vectors.end()) return results;

    // Score against all stored signatures
    std::vector<std::pair<double, int>> scored;
    for (const auto& [sid, bv] : impl_->bow_vectors) {
        if (sid == query_sig_id) continue;
        double s = impl_->vocab->score(it_q->second, bv);
        scored.emplace_back(s, sid);
    }

    std::sort(scored.begin(), scored.end(), std::greater<>());
    int count = 0;
    for (const auto& [s, sid] : scored) {
        if (count >= top_k) break;
        results[sid] = s;
        ++count;
    }
    return results;
}

std::map<int, double> BowDatabase::computeLikelihoods(int query_sig_id) const {
    std::map<int, double> likelihoods;
    if (!vocab_loaded_) return likelihoods;

    auto it_q = impl_->bow_vectors.find(query_sig_id);
    if (it_q == impl_->bow_vectors.end()) return likelihoods;

    for (const auto& [sid, bv] : impl_->bow_vectors) {
        if (sid == query_sig_id) continue;
        likelihoods[sid] = impl_->vocab->score(it_q->second, bv);
    }
    return likelihoods;
}

void BowDatabase::removeSignature(int sig_id) {
    impl_->bow_vectors.erase(sig_id);
    impl_->feat_vectors.erase(sig_id);
}

}  // namespace vo
