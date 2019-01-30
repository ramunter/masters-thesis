def move_right(env):
    """
    This is the optimal policy in the Osband example. Used to test env.
    """

    move_right = MoveRight()

    _ = env.reset()
    action = move_right.policy()
    done = False

    while not done:

        # Perform step
        _, _, done, _ = env.step(action)
        # Calculate Q-values
        action = move_right.policy()


class MoveRight():
    # Class to test that things are working as expected
    def policy(self):
        return 1
