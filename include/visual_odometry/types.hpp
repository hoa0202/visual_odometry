#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vo {
struct ProcessedImages {
    cv::Mat gray;
    cv::Mat enhanced;
    cv::Mat denoised;
    cv::Mat masked;
};

struct CameraParams {
    double fx{0.0};  // 초점 거리 x
    double fy{0.0};  // 초점 거리 y
    double cx{0.0};  // 주점 x
    double cy{0.0};  // 주점 y
    int width{0};    // 이미지 너비
    int height{0};   // 이미지 높이
    
    // 카메라 행렬 반환
    cv::Mat getCameraMatrix() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0,0) = fx;
        K.at<double>(1,1) = fy;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;
        return K;
    }
};

struct Features {
    std::vector<cv::KeyPoint> keypoints;  // 특징점 좌표
    cv::Mat descriptors;                  // 특징점 디스크립터
    cv::Mat visualization;               // 시각화용 이미지 (선택적)
};

// 매칭 결과를 저장하는 구조체 추가
struct FeatureMatches {
    std::vector<cv::DMatch> matches;           // 매칭 결과
    std::vector<cv::Point2f> prev_points;      // 이전 프레임의 매칭점 좌표
    std::vector<cv::Point2f> curr_points;      // 현재 프레임의 매칭점 좌표
    
    // 메모리 예약을 위한 메서드 추가
    void reserve(size_t size) {
        matches.reserve(size);
        prev_points.reserve(size);
        curr_points.reserve(size);
    }
    
    // 매칭 결과 초기화
    void clear() {
        matches.clear();
        prev_points.clear();
        curr_points.clear();
    }
    
    // 매칭 개수 반환
    size_t size() const {
        return matches.size();
    }
    
    // 매칭이 비어있는지 확인
    bool empty() const {
        return matches.empty();
    }
};
} // namespace vo 