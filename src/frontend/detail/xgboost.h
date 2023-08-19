/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file xgboost.h
 * \brief Helper functions for loading XGBoost models
 * \author William Hicks
 */
#ifndef SRC_FRONTEND_DETAIL_XGBOOST_H_
#define SRC_FRONTEND_DETAIL_XGBOOST_H_

#include <cmath>
#include <string>
#include <vector>

namespace treelite {

namespace details::xgboost {

struct ProbToMargin {
  static float Sigmoid(float global_bias) {
    return -std::log(1.0f / global_bias - 1.0f);
  }
  static float Exponential(float global_bias) {
    return std::log(global_bias);
  }
};

// set correct prediction transform function, depending on objective function
void SetPredTransform(std::string const& objective_name, ModelParam* param);

// Transform the global bias parameter from probability into margin score
void TransformGlobalBiasToMargin(ModelParam* param);

}  // namespace details::xgboost
}  // namespace treelite

#endif  // SRC_FRONTEND_DETAIL_XGBOOST_H_
