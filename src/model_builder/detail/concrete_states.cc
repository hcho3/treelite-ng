#include "state_machine.h"
/*!
 * Copyright (c) 2023 by Contributors
 * \file concrete_states.cc
 * \brief Concrete state classes
 * \author Hyunsu Cho
 */

#include "./concrete_states.h"

namespace treelite::model_builder::detail {

void ExpectTree::Enter(StateMachine* machine) const {}

void ExpectTree::Toggle(StateMachine* machine) const {}

void ExpectTree::Exit(StateMachine* machine) const {}

State& ExpectTree::GetInstance() {
  static ExpectTree singleton;
  return singleton;
}

void ExpectNode::Enter(StateMachine* machine) const {}

void ExpectNode::Toggle(StateMachine* machine) const {}

void ExpectNode::Exit(StateMachine* machine) const {}

State& ExpectNode::GetInstance() {
  static ExpectNode singleton;
  return singleton;
}

void ExpectDetail::Enter(StateMachine* machine) const {}

void ExpectDetail::Toggle(StateMachine* machine) const {}

void ExpectDetail::Exit(StateMachine* machine) const {}

State& ExpectDetail::GetInstance() {
  static ExpectDetail singleton;
  return singleton;
}

void NodeComplete::Enter(StateMachine* machine) const {}

void NodeComplete::Toggle(StateMachine* machine) const {}

void NodeComplete::Exit(StateMachine* machine) const {}

State& NodeComplete::GetInstance() {
  static NodeComplete singleton;
  return singleton;
}

}  // namespace treelite::model_builder::detail
