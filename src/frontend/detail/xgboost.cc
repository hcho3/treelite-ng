/*!
 * Copyright (c) 2023 by Contributors
 * \file xgboost.cc
 * \brief Utility functions for XGBoost frontend
 * \author Hyunsu Cho
 */
#include "./xgboost.h"

#include <treelite/logging.h>
#include <treelite/tree.h>

#include <string>
#include <vector>

namespace treelite::frontend::detail::xgboost {

// set correct prediction transform function, depending on objective function
std::string GetPredTransform(std::string const& objective_name) {
  const std::vector<std::string> exponential_objectives{
      "count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft"};
  if (objective_name == "multi:softmax") {
    return "max_index";
  } else if (objective_name == "multi:softprob") {
    return "softmax";
  } else if (objective_name == "reg:logistic" || objective_name == "binary:logistic") {
    return "sigmoid";
  } else if (std::find(
                 exponential_objectives.cbegin(), exponential_objectives.cend(), objective_name)
             != exponential_objectives.cend()) {
    return "exponential";
  } else if (objective_name == "binary:hinge") {
    return "hinge";
  } else if (objective_name == "reg:squarederror" || objective_name == "reg:linear"
             || objective_name == "reg:squaredlogerror" || objective_name == "reg:pseudohubererror"
             || objective_name == "binary:logitraw" || objective_name == "rank:pairwise"
             || objective_name == "rank:ndcg" || objective_name == "rank:map") {
    return "identity";
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized XGBoost objective: " << objective_name;
    return "";
  }
}

double TransformBaseScoreToMargin(std::string const& pred_transform, double base_score) {
  if (pred_transform == "sigmoid") {
    return ProbToMargin::Sigmoid(base_score);
  } else if (pred_transform == "exponential") {
    return ProbToMargin::Exponential(base_score);
  } else {
    return base_score;
  }
}

}  // namespace treelite::frontend::detail::xgboost
