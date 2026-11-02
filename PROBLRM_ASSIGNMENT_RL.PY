import numpy as np

ACTIONS = ["up", "down", "left", "right"]
GAMMA = 1.0
THETA = 1e-8
REWARD = -1.0


# Original Gridworld: states 0 and 15 are terminal
def step(state, action):    ZeroDivisionError
    if state in [0, 15]:
        return state, 0

    row, col = divmod(state, 4)

    if action == "up":
        row = max(0, row - 1)
    elif action == "down":
        row = min(3, row + 1)
    elif action == "left":
        col = max(0, col - 1)
    elif action == "right":
        col = min(3, col + 1)

    next_state = row * 4 + col
    return next_state, REWARD


# Policy evaluation for the equiprobable random policy
def evaluate_policy(extended=False, change_13=False):
    # Original grid has states 0-15.
    # Extended grid renumbers the original terminal 15 as 16
    # and adds the new state 15.
    n_states = 17 if extended else 16
    V = np.zeros(n_states)

    while True:
        V_new = V.copy()
        delta = 0.0

        last_state = 15 if extended else 14

        for s in range(1, last_state + 1):
            value = 0.0

            for action in ACTIONS:

                # Transitions from the new state 15
                if extended and s == 15:
                    next_s = {
                        "left": 12,
                        "up": 13,
                        "right": 14,
                        "down": 15
                    }[action]
                    reward = REWARD

                # Exercise 4.2: change down from state 13
                elif extended and s == 13 and action == "down" and change_13:
                    next_s = 15
                    reward = REWARD

                else:
                    next_s, reward = step(s, action)

                    # In the extended grid, old terminal 15 becomes 16.
                    if extended and next_s == 15:
                        next_s = 16

                value += 0.25 * (
                    reward + GAMMA * V[next_s]
                )

            V_new[s] = value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < THETA:
            break

    return V


# Value iteration for optimal values
def value_iteration():
    V = np.zeros(16)

    while True:
        V_new = V.copy()
        delta = 0.0

        for s in range(1, 15):
            action_values = []

            for action in ACTIONS:
                next_s, reward = step(s, action)
                action_values.append(
                    reward + GAMMA * V[next_s]
                )

            V_new[s] = max(action_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < THETA:
            break

    return V


# ---------------- MAIN PROGRAM ----------------

# Exercise 4.1
V = evaluate_policy()

q_11_down = step(11, "down")[1] + GAMMA * V[step(11, "down")[0]]
q_7_down = step(7, "down")[1] + GAMMA * V[step(7, "down")[0]]


print("q_pi(11, down) =", round(q_11_down, 2))
print("q_pi(7, down)  =", round(q_7_down, 2))

# Exercise 4.2 - Case 1
V_case1 = evaluate_policy(extended=True, change_13=False)

# Exercise 4.2 - Case 2
V_case2 = evaluate_policy(extended=True, change_13=True)


print("Case 1: v_pi(15) =", round(V_case1[15], 2))
print("Case 2: v_pi(15) =", round(V_case2[15], 2))

# Value iteration
V_star = value_iteration()

print("\nOPTIMAL VALUES USING VALUE ITERATION")
print(np.round(V_star.reshape(4, 4), 2))
