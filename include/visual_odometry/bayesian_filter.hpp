#pragma once

#include <map>
#include <vector>

namespace vo {

// RTAB-Map style Bayesian loop closure filter.
// Maintains a posterior distribution over all locations (signatures)
// and a virtual "new place" hypothesis. Accepts a loop closure when
// the same hypothesis wins for `temporal_consistency` consecutive updates
// and its posterior exceeds `threshold`.
class BayesianFilter {
public:
    struct Hypothesis {
        int sig_id{-1};
        double probability{0.0};
    };

    explicit BayesianFilter(double threshold = 0.55, int temporal_consistency = 3);

    // Add a new location (signature) to the state space
    void addLocation(int sig_id);

    // Remove a location (when transferred to LTM and pruned)
    void removeLocation(int sig_id);

    // Set neighbor relationships for transition model
    void setNeighbors(int sig_id, const std::vector<int>& neighbor_ids);

    // Update posterior given likelihoods {sig_id → P(z|S_i)} from BoW query
    // Returns accepted loop hypothesis (sig_id >= 0) or -1 if no loop
    Hypothesis update(const std::map<int, double>& likelihoods, int current_sig_id);

    // Reset filter state
    void reset();

    // Get current posterior for debugging
    const std::map<int, double>& posterior() const { return posterior_; }
    double virtualPlacePosterior() const { return virtual_posterior_; }

private:
    double threshold_;
    int temporal_consistency_;

    // Posterior over all locations + virtual place
    std::map<int, double> posterior_;
    double virtual_posterior_{1.0};  // starts with all mass on "new place"

    // Transition model: neighbors for each sig
    std::map<int, std::vector<int>> neighbors_;

    // Temporal consistency tracking
    int last_winner_{-1};
    int consecutive_wins_{0};

    // Virtual place prior (RTAB-Map default)
    static constexpr double kVirtualPlacePrior = 0.9;
    static constexpr double kPredictionLC = 0.04;

    void normalize();
    void predict();
};

}  // namespace vo
