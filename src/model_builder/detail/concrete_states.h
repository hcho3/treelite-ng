/*!
 * Copyright (c) 2023 by Contributors
 * \file concrete_states.h
 * \brief Concrete state classes
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_BUILDER_DETAIL_CONCRETE_STATES_H_
#define SRC_MODEL_BUILDER_DETAIL_CONCRETE_STATES_H_

#include "./state_machine.h"

namespace treelite::model_builder::detail {

class ExpectTree : public State {
 public:
  void Enter(StateMachine* machine) const override;
  void Toggle(StateMachine* machine) const override;
  void Exit(StateMachine* machine) const override;
  static State& GetInstance();
};

class ExpectNode : public State {
 public:
  void Enter(StateMachine* machine) const override;
  void Toggle(StateMachine* machine) const override;
  void Exit(StateMachine* machine) const override;
  static State& GetInstance();
};

class ExpectDetail : public State {
 public:
  void Enter(StateMachine* machine) const override;
  void Toggle(StateMachine* machine) const override;
  void Exit(StateMachine* machine) const override;
  static State& GetInstance();
};

class NodeComplete : public State {
 public:
  void Enter(StateMachine* machine) const override;
  void Toggle(StateMachine* machine) const override;
  void Exit(StateMachine* machine) const override;
  static State& GetInstance();
};

}  // namespace treelite::model_builder::detail

#endif  // SRC_MODEL_BUILDER_DETAIL_CONCRETE_STATES_H_
