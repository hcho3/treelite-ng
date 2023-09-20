/*!
 * Copyright (c) 2023 by Contributors
 * \file sklearn.cc
 * \author Hyunsu Cho
 * \brief C API for scikit-learn loader functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

int TreeliteLoadSKLearnRandomForestRegressor(int n_estimators, int n_features, int n_targets,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadRandomForestRegressor(n_estimators, n_features,
      n_targets, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnIsolationForest(int n_estimators, int n_features, int64_t const* node_count,
    int64_t const** children_left, int64_t const** children_right, int64_t const** feature,
    double const** threshold, double const** value, int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c,
    TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadIsolationForest(n_estimators, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, ratio_c);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestClassifier(int n_estimators, int n_features, int n_targets,
    int32_t const* n_classes, int64_t const* node_count, int64_t const** children_left,
    int64_t const** children_right, int64_t const** feature, double const** threshold,
    double const** value, int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadRandomForestClassifier(n_estimators, n_features,
      n_targets, n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingRegressor(int n_iter, int n_features,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadGradientBoostingRegressor(n_iter, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, double const** value,
    int64_t const** n_node_samples, double const** weighted_n_node_samples, double const** impurity,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadGradientBoostingClassifier(n_iter, n_features,
      n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingRegressor(int n_iter, int n_features,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, int8_t const** default_left,
    double const** value, int64_t const** n_node_samples, double const** gain,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadHistGradientBoostingRegressor(n_iter,
      n_features, node_count, children_left, children_right, feature, threshold, default_left,
      value, n_node_samples, gain, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    int64_t const* node_count, int64_t const** children_left, int64_t const** children_right,
    int64_t const** feature, double const** threshold, int8_t const** default_left,
    double const** value, int64_t const** n_node_samples, double const** gain,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadHistGradientBoostingClassifier(n_iter,
      n_features, n_classes, node_count, children_left, children_right, feature, threshold,
      default_left, value, n_node_samples, gain, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
