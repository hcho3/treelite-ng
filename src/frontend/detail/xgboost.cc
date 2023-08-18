/*!
 * Copyright (c) 2023 by Contributors
 * \file xgboost.cc
 * \brief Utility functions for XGBoost frontend
 * \author Hyunsu Cho
 */
#include "./xgboost.h"

#include <treelite/tree.h>
#include <treelite/logging.h>

#include <string>
#include <vector>

namespace {

inline void SetPredTransformString(char const* value, treelite::ModelParam* param) {
  std::strncpy(param->pred_transform, value, sizeof(param->pred_transform));
}

}  // anonymous namespace

namespace treelite::details::xgboost {

// set correct prediction transform function, depending on objective function
void SetPredTransform(std::string const& objective_name, ModelParam *param) {
  const std::vector<std::string> exponential_objectives{
    "count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft"};
  if (objective_name == "multi:softmax") {
    SetPredTransformString("max_index", param);
  } else if (objective_name == "multi:softprob") {
    SetPredTransformString("softmax", param);
  } else if (objective_name == "reg:logistic" || objective_name == "binary:logistic") {
    SetPredTransformString("sigmoid", param);
    param->sigmoid_alpha = 1.0f;
  } else if (std::find(
      exponential_objectives.cbegin(), exponential_objectives.cend(), objective_name)
             != exponential_objectives.cend()) {
    SetPredTransformString("exponential", param);
  } else if (objective_name == "binary:hinge") {
    SetPredTransformString("hinge", param);
  } else if (objective_name == "reg:squarederror" || objective_name == "reg:linear"
             || objective_name == "reg:squaredlogerror" || objective_name == "reg:pseudohubererror"
             || objective_name == "binary:logitraw" || objective_name == "rank:pairwise"
             || objective_name == "rank:ndcg" || objective_name == "rank:map") {
    SetPredTransformString("identity", param);
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized XGBoost objective: " << objective_name;
  }
}

// Transform the global bias parameter from probability into margin score
void TransformGlobalBiasToMargin(ModelParam *param) {
  std::string bias_transform{param->pred_transform};
  if (bias_transform == "sigmoid") {
    param->global_bias = ProbToMargin::Sigmoid(param->global_bias);
  } else if (bias_transform == "exponential") {
    param->global_bias = ProbToMargin::Exponential(param->global_bias);
  }
}

}  // treelite::details::xgboost
