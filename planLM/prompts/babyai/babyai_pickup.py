class PromptBabyAIPickup:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        prompt = \
            '[Goal] pick up the box. ' \
            '[Initial State] Room 1 has red ball, green key, agent. ' \
            'Room 2 has blue box. ' \
            'The door connecting Room 1 and Room 2 is locked. ' \
            'The red ball is blocking the door. ' \
            '[Step 1] pick up red ball. ' \
            '[Step 2] drop ball in void. ' \
            '[Step 3] pick up green key. ' \
            '[Step 4] toggle green door. ' \
            '[Step 5] drop key in void. ' \
            '[Step 6] pick up blue box. ' \
            '[Step 7] done picking up. '

        prompt = \
            f'{prompt}' \
            '[Goal] pick up the purple box. ' \
            '[Initial State] Room 1 has yellow key, agent. '  \
            'Room 2 has purple box. ' \
            'The door connecting Room 1 and Room 2 is locked. ' \
            '[Step 1] pick up yellow key. ' \
            '[Step 2 toggle yellow door. ' \
            '[Step 3] drop key in void. ' \
            '[Step 4] pick up purple box. ' \
            '[Step 5] done picking up. '

        prompt = \
            f'{prompt}' \
            '[Goal] pick up the ball. ' \
            '[Initial State] Room 1 has yellow ball. ' \
            'Room 2 has agent, grey key. ' \
            'Room 3 has blue key. ' \
            'The door connecting Room 1 and Room 2 is locked. ' \
            'The door connecting Room 2 and Room 3 is locked. ' \
            '[Step 1] pick up grey key. ' \
            '[Step 2] toggle grey door. ' \
            '[Step 3] drop key in void. ' \
            '[Step 4] pick up blue key. ' \
            '[Step 5] toggle blue door. ' \
            '[Step 6] drop key in void. ' \
            '[Step 7] pick up yellow ball. ' \
            '[Step 8] done picking up. '

        return prompt