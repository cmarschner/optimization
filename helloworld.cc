#include "ceres/ceres.h"
#include "glog/logging.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

int kNumPoints = 5;

const Eigen::Vector3d kXY[] = {
   Eigen::Vector3d(1, 8, 0),
   Eigen::Vector3d(1, 5, 0),
   Eigen::Vector3d(2, 4, 0),
   Eigen::Vector3d(6, 4, 0),
   Eigen::Vector3d(10.4, 3, 0)
};

double totalLength() {
   double length = 0;
   for (int i = 1; i < kNumPoints; ++i) {
     length += (kXY[i] - kXY[i - 1]).norm();
   }
   return length;
}

std::vector<Eigen::Vector3d> points;

void initPoints() {
  double length = totalLength();
  double step = 0.5; 
  int numPoints = 1 + length / step; // plus one endpoint
  std::cout << "length " << length << " step " << step << " numPoints " << numPoints;
  points.push_back(Eigen::Vector3d(kXY[0]));
  int curPoint = 0;
  double curVecPart = 0;
  Eigen::Vector3d curVec = kXY[curPoint + 1] - kXY[curPoint];
  double remaining = 0;
  for (int i = 1; i < numPoints; ++i) {
    std::cout << "point " << i << std::endl;
    for (double curPart = 0; curPart < 1;) {
      remaining = curVec.norm() * (1 - curVecPart);
      std::cout << "remaining " << remaining << " curpart " << curPart << " 1-curPart*step " << (1-curPart)*step << std::endl;
      if (remaining < (1 - curPart) * step) {
        curPart += remaining / step;
        curPoint++;
        curVec = kXY[curPoint + 1] - kXY[curPoint];
        std::cout << "new vector " << curPoint << std::endl;
        curVecPart = 0;
      }
      else {
        curVecPart += ((1 - curPart) * step) / curVec.norm();
        curPart = 1;
        Eigen::Vector3d res(kXY[curPoint] + curVecPart * curVec);
        std::cout << "Found " << res[0] << " " << res[1] << std::endl;
        points.push_back(kXY[curPoint] + curVecPart * curVec);
      }
    }
  }
  if (remaining > 0) 
    points.push_back(kXY[kNumPoints - 1]);
}

void display() {
  // The variable to solve for with its initial value.
  cv::Mat img(1100,1100, CV_8UC3, cv::Scalar(40,40,80));
  for (int i = 0; i < kNumPoints; ++i) {
    Eigen::Vector3d vec = kXY[i] * 100;
    cv::rectangle( img, cv::Point( vec[0] - 3, vec[1] - 3 ), cv::Point( vec[0] + 3,vec[1] + 3), cv::Scalar( 255, 55, 255 ), CV_FILLED, 4 );
  }
  Eigen::Vector3d p1 = points[0] * 100;

  cv::rectangle( img, cv::Point( p1[0] - 2, p1[1] - 2 ), cv::Point( p1[0] + 2, p1[1] + 2), cv::Scalar( 100, 200, 100 ), CV_FILLED, 4 );
  for (int i = 1; i < points.size(); ++i) {
    Eigen::Vector3d p1 = points[i - 1] * 100;
    Eigen::Vector3d p2 = points[i] * 100;
    cv::line( img, cv::Point (p1[0], p1[1]), cv::Point (p2[0], p2[1]), cv::Scalar( 55, 255, 55 ), 1);
    cv::rectangle( img, cv::Point( p1[0] - 2, p1[1] - 2 ), cv::Point( p1[0] + 2, p1[1] + 2), cv::Scalar( 55, 55, 255 ), CV_FILLED, 4 );
  }
  cv::imshow("foo", img);
  cv::waitKey(0);

}

bool SolveWithFullReport(ceres::Solver::Options options,
                         ceres::Problem* problem,
                         bool dynamic_sparsity) {
  options.dynamic_sparsity = dynamic_sparsity;
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  std::cout << "####################" << std::endl;
  std::cout << "dynamic_sparsity = " << dynamic_sparsity << std::endl;
  std::cout << "####################" << std::endl;
  std::cout << summary.FullReport() << std::endl;
  return summary.termination_type == ceres::CONVERGENCE;
}


struct EqualDistanceCostFunctor {
   EqualDistanceCostFunctor(int i) :  
     i_(i)
   {}

   template <typename T>
   bool operator()(const T* point0,const T* const point1, const T* const point2, T* residual) const {
     Eigen::Map<const Eigen::Matrix<T, 3, 1> > p0(point0);
     Eigen::Map<const Eigen::Matrix<T, 3, 1> > p1(point1);
     Eigen::Map<const Eigen::Matrix<T, 3, 1> > p2(point2);

     const Eigen::Matrix<T, 3, 1> distance_vector = (p2 + p0) / 2. - p1; // p2_.cast<T>() - p1;
     residual[0] = distance_vector[0];
     residual[1] = distance_vector[1];
     residual[2] = distance_vector[2];
     // std::cout << "equald / distance " << i_ << ":" << std::endl << distance_vector << std::endl;
     return true;
   }
   int i_;
};

struct FollowTrajectoryCostFunctor {
   FollowTrajectoryCostFunctor(int i, int j) :  
     i_(i), j_(j)
   {}

   template <typename T>
   bool operator()(const T* point,T* residual) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1> > p(point);
      T distance(0); //Eigen::Matrix<T, 3, 1> distance;
      T d = (p - kXY[j_].cast<T>()).norm();
      if (d > T(1e-6)) {
        distance += d;
      }
      distance /= T(kNumPoints);
      std::cout << " at i=" << i_ <<  " j=" << j_ << std::endl << distance << std::endl;
      residual[0] = distance;
      std::cout << "followt / distance " << i_ << ":" << std::endl << distance << std::endl;
      return true;
   }
   int i_, j_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // Problem configuration.
  initPoints();
  display();
  for (int n = 1; n < 10; ++n) {
   ceres::Problem problem;
   for (int i = 0; i < points.size(); ++i) {
    problem.AddParameterBlock(points[i].data(), 3);
   }
   for (int i = 1; i < points.size() - 1; ++i) {
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<EqualDistanceCostFunctor, 3, 3, 3, 3>(new EqualDistanceCostFunctor(i)), nullptr, points[i - 1].data(), points[i].data(), points[i + 1].data());
    for (int j = 0; j < kNumPoints; ++j) {
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FollowTrajectoryCostFunctor, 1, 3>(new FollowTrajectoryCostFunctor(i, j)), new ceres::HuberLoss(5), points[i].data());
    }
   }	  
   problem.SetParameterBlockConstant(points[0].data());
   problem.SetParameterBlockConstant(points[points.size() - 1].data());
   ceres::Solver::Options options;
   options.max_num_iterations = 100;
   options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
   SolveWithFullReport(options, &problem, false);
   display();
  }

  return 0;
}


