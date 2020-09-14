import torch
import torch.nn.functional as F

class Agent():
    def __init__(self, environment, agent_index, num_actions, state_estimator, opponent_action_estimator, value_estimator, state_estimator_optimizer, opponent_action_estimator_optimizer, value_estimator_optimizer, exploration=False, discount=0.99):
        self.environment = environment
        self.agent_index = agent_index
        self.num_actions = num_actions
        self.state_estimator = state_estimator
        self.opponent_action_estimator = opponent_action_estimator
        self.value_estimator = value_estimator
        self.state_estimator_optimizer = state_estimator_optimizer
        self.opponent_action_estimator_optimizer = opponent_action_estimator_optimizer
        self.value_estimator_optimizer = value_estimator_optimizer
        self.exploration = exploration
        self.discount = discount
        self.memory = []

    def get_action(state):
        with torch.no_grad():
            values = []
            for action in range(self.num_actions):
                next_state = self.state_estimator(state.unsqueeze(0), torch.tensor(action, device=state.device).unsqueeze(0))
                opponent_action = torch.argmax(self.opponent_action_estimator(next_state))
                next_next_state = self.state_estimator(next_state, opponent_action)
                value = self.value_estimator(next_next_state)
                values.append(value[0])
            values = torch.stack(values)
            if self.exploration:
                probabilities = torch.softmax(values, dim=-1)
                action = torch.multinomial(probabilities, 1)[0]
            else:
                action = torch.argmax(values)
        return action

    def step(state, opponent):
        action = self.get_action(state)
        next_state, reward, terminal = self.environment.step(state, action, self.agent_index)
        if not terminal:
            opponent_action = opponent.get_action(next_state)
            next_next_state = self.environment.step(next_state, opponent_action, opponent.agent_index)
        self.memory.append((state, action, next_state, opponent_action, next_next_state, reward))
        return terminal

    def get_data():
        values = []
        for state, action, next_state, opponent_action, next_next_state, reward in reversed(self.memory):
            value = reward
            if values:
                value += self.discount * values[-1]
            values.append(value)
        data = []
        for (state, action, next_state, opponent_action, next_next_state, reward), value in zip(self.memory, reversed(values)):
            data.append(state, action, next_state, opponent_action, next_next_state, value)
        self.memory = []
        return data

    def update(data):
        state, action, next_state, opponent_action, next_next_state, value = [torch.stack(x) for x in zip(*data)]
        all_states = torch.cat([state, next_state])
        all_actions = torch.cat([action, opponent_action])
        all_next_states = torch.cat([next_state, next_next_state])
        self.state_estimator_optimizer.zero_grad()
        self.opponent_action_estimator_optimizer.zero_grad()
        self.value_estimator_optimizer.zero_grad()
        state_estimate = self.state_estimator(all_states, all_actions)
        opponent_action_estimate = self.opponent_action_estimator(next_state)
        value_estimate = self.value_estimator(next_next_state)
        state_loss = F.mse_loss(state_estimate, all_next_states)
        opponent_action_loss = F.cross_entropy(opponent_action_estimate, opponent_action)
        value_loss = F.mse_loss(value_estimate, value)
        state_loss.backward()
        opponent_action_loss.backward()
        value_loss.backward()
        self.state_estimator_optimizer.step()
        self.opponent_action_estimator_optimizer.step()
        self.value_estimator_optimizer.step()
