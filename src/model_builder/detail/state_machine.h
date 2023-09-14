/*!
 * Copyright (c) 2023 by Contributors
 * \file state_machine.h
 * \brief State machine for the model builder API
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_BUILDER_DETAIL_STATE_MACHINE_H_
#define SRC_MODEL_BUILDER_DETAIL_STATE_MACHINE_H_

namespace treelite::model_builder::detail {

class StateMachine;

class State {
 public:
  virtual void Enter(StateMachine* machine) const = 0;
  virtual void Toggle(StateMachine* machine) const = 0;
  virtual void Exit(StateMachine* machine) const = 0;
  virtual ~State() = default;
};

class StateMachine {
 public:
  StateMachine();
  State* GetCurrentState() const {
    return current_state_;
  }
  void Toggle();
  void SetState(State& new_state);

 private:
  State* current_state_;
};

}  // namespace treelite::model_builder::detail

#endif  // SRC_MODEL_BUILDER_DETAIL_STATE_MACHINE_H_
