/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file pred_transform.h
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */

#ifndef SRC_GTIL_PRED_TRANSFORM_H_
#define SRC_GTIL_PRED_TRANSFORM_H_

#include <string>

namespace treelite {

class Model;

namespace gtil {

template <typename InputT>
using PredTransformFunc = void (*)(treelite::Model const&, std::uint32_t, InputT*);

template <typename InputT>
PredTransformFunc<InputT> GetPredTransformFunc(std::string const& name);

extern template PredTransformFunc<float> GetPredTransformFunc(std::string const&);
extern template PredTransformFunc<double> GetPredTransformFunc(std::string const&);

}  // namespace gtil
}  // namespace treelite

#endif  // SRC_GTIL_PRED_TRANSFORM_H_
