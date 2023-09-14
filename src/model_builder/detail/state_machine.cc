/*!
 * Copyright (c) 2023 by Contributors
 * \file state_machine.cc
 * \brief State machine for the model builder API
 * \author Hyunsu Cho
 */

#include "./state_machine.h"

namespace treelite::model_builder::detail {

StateMachine::StateMachine() {
  // TODO(hcho3): Set the initial state here
}

void StateMachine::Toggle() {
  // Delegate the task of determining the next state to the current state
  current_state_->Toggle(this);
}

void StateMachine::SetState(State& new_state) {
  current_state_->Exit(this);  // do something before we change state
  current_state_ = &new_state;  // change state
  current_state_->Enter(this);  // do something after we change state
}

}  // namespace treelite::model_builder::detail
