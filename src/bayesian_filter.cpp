#include "visual_odometry/bayesian_filter.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace vo {

BayesianFilter::BayesianFilter(double threshold, int temporal_consistency)
    : threshold_(threshold), temporal_consistency_(temporal_consistency) {}

void BayesianFilter::addLocation(int sig_id) {
    if (posterior_.count(sig_id)) return;
    // New location gets a small slice of the virtual place mass
    double share = virtual_posterior_ * 0.01;
    posterior_[sig_id] = share;
    virtual_posterior_ -= share;
    if (virtual_posterior_ < 0.01) virtual_posterior_ = 0.01;
}

void BayesianFilter::removeLocation(int sig_id) {
    auto it = posterior_.find(sig_id);
    if (it != posterior_.end()) {
        virtual_posterior_ += it->second;
        posterior_.erase(it);
    }
    neighbors_.erase(sig_id);
}

void BayesianFilter::setNeighbors(int sig_id, const std::vector<int>& neighbor_ids) {
    neighbors_[sig_id] = neighbor_ids;
}

BayesianFilter::Hypothesis BayesianFilter::update(
    const std::map<int, double>& likelihoods, int current_sig_id) {

    Hypothesis result;
    if (posterior_.empty()) return result;

    // --- Prediction step (transition model) ---
    predict();

    // --- Update step (Bayes rule) ---
    // P(S_i | z) ∝ P(z | S_i) * P(S_i)  [predicted prior]
    // Virtual place: likelihood = average of all likelihoods (mildly informative)
    double avg_likelihood = 0.0;
    int lk_count = 0;
    for (const auto& [sid, lk] : likelihoods) {
        avg_likelihood += lk;
        ++lk_count;
    }
    avg_likelihood = (lk_count > 0) ? avg_likelihood / lk_count : 0.01;

    // Update each location's posterior
    for (auto& [sid, prior] : posterior_) {
        auto it = likelihoods.find(sid);
        double lk = (it != likelihoods.end()) ? it->second : 1e-6;
        // Skip current and very recent neighbors
        if (sid == current_sig_id) { prior *= 1e-6; continue; }
        prior *= lk;
    }

    // Virtual place update
    virtual_posterior_ *= avg_likelihood;

    normalize();

    // --- Find winner ---
    int best_id = -1;
    double best_prob = 0.0;
    for (const auto& [sid, prob] : posterior_) {
        if (sid == current_sig_id) continue;
        if (prob > best_prob) { best_prob = prob; best_id = sid; }
    }

    // --- Temporal consistency check ---
    if (best_id >= 0 && best_prob > threshold_) {
        if (best_id == last_winner_) {
            consecutive_wins_++;
        } else {
            last_winner_ = best_id;
            consecutive_wins_ = 1;
        }

        if (consecutive_wins_ >= temporal_consistency_) {
            result.sig_id = best_id;
            result.probability = best_prob;
            last_winner_ = -1;
            consecutive_wins_ = 0;
            // Diffuse posterior after acceptance to prevent re-trigger
            for (auto& [sid, p] : posterior_) p = 1e-4;
            virtual_posterior_ = 1.0;
            normalize();
        }
    } else {
        if (best_id != last_winner_) {
            last_winner_ = best_id;
            consecutive_wins_ = (best_id >= 0) ? 1 : 0;
        }
    }

    return result;
}

void BayesianFilter::predict() {
    // RTAB-Map transition model:
    // - Each location spreads probability to its neighbors
    // - Virtual place gets a constant prior pull
    std::map<int, double> predicted;

    for (const auto& [sid, prob] : posterior_) {
        // Self-transition: keep most mass on itself
        double self_mass = prob * (1.0 - kPredictionLC);
        predicted[sid] += self_mass;

        // Spread to neighbors
        auto nit = neighbors_.find(sid);
        if (nit != neighbors_.end() && !nit->second.empty()) {
            double spread = prob * kPredictionLC / nit->second.size();
            for (int nbr : nit->second) {
                if (posterior_.count(nbr))
                    predicted[nbr] += spread;
            }
        }
    }

    // Virtual place attraction
    double vp_pull = kVirtualPlacePrior * 0.01;
    for (auto& [sid, p] : predicted)
        p *= (1.0 - vp_pull);
    virtual_posterior_ = virtual_posterior_ * (1.0 - vp_pull) + vp_pull;

    posterior_ = predicted;
    normalize();
}

void BayesianFilter::normalize() {
    double sum = virtual_posterior_;
    for (const auto& [sid, p] : posterior_) sum += p;
    if (sum < 1e-12) {
        // Degenerate — reset to uniform
        virtual_posterior_ = 0.5;
        double each = 0.5 / std::max(1, static_cast<int>(posterior_.size()));
        for (auto& [sid, p] : posterior_) p = each;
        return;
    }
    double inv = 1.0 / sum;
    for (auto& [sid, p] : posterior_) p *= inv;
    virtual_posterior_ *= inv;
}

void BayesianFilter::reset() {
    posterior_.clear();
    virtual_posterior_ = 1.0;
    neighbors_.clear();
    last_winner_ = -1;
    consecutive_wins_ = 0;
}

}  // namespace vo
