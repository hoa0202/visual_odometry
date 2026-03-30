#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace DBoW2 {
class BowVector;
class FeatureVector;
}

namespace vo {

class BowDatabase {
public:
    explicit BowDatabase(const std::string& vocab_path = "");
    ~BowDatabase();

    bool isReady() const { return vocab_loaded_; }

    // Convert ORB descriptors (cv::Mat rows) → BoW vector ID
    // Returns internal bow_id for later similarity queries
    int addSignature(int sig_id, const cv::Mat& descriptors);

    // BoW similarity between two signatures (0.0 ~ 1.0)
    double score(int sig_id_a, int sig_id_b) const;

    // Query: get similarity scores of sig_id against all stored signatures
    // Returns map<sig_id, score> sorted by score descending, top_k results
    std::map<int, double> query(int query_sig_id, int top_k = 10) const;

    // Raw similarity between a query and all stored (for Bayesian filter)
    // Returns map<sig_id, likelihood>
    std::map<int, double> computeLikelihoods(int query_sig_id) const;

    void removeSignature(int sig_id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool vocab_loaded_{false};
};

}  // namespace vo
