class PromptRavensPutBlockInBowl:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):

        prompt = \
            '[Goal] put the red blocks in green bowls. ' \
            '[Initial State] There is a red block 1, red block 2, yellow block 1, ' \
            'blue block 1, red block 3, green bowl 1, blue bowl 1, ' \
            'blue bowl 2, green bowl 2, green bowl 3. ' \
            f'[Step 1] place red block 1 in green bowl 1. ' \
            f'[Step 2] place red block 2 in green bowl 2. ' \
            f'[Step 3] place red block 3 in green bowl 3. ' \
            f'[Step 4] done placing blocks in bowls. '

        prompt = \
            f'{prompt}' \
            '[Goal] put the green blocks in blue bowls. '\
            '[Initial State] There is a green block 1, red block 1, green block 2, blue block 1, ' \
            'blue block 2, blue bowl 1, blue bowl 2, red bowl 1. ' \
            f'[Step 1] place green block 1 in blue bowl 1. ' \
            f'[Step 2] place green block 2 in blue bowl 2. ' \
            f'[Step 3] done placing blocks in bowls. '

        # prompt = \
        #     f'{prompt}' \
        #     '[Goal] put the blue blocks in yellow bowls. ' \
        #     '[Initial State] There is a yellow block 1, yellow bowl 1, yellow bowl 2, ' \
        #     'yellow bowl 3, green block 1, green bowl 1, blue block 1, ' \
        #     'blue block 2, blue block 3. ' \
        #     f'[Step 1] place blue block 1 in yellow bowl 1. ' \
        #     f'[Step 2] place blue block 2 in yellow bowl 2. ' \
        #     f'[Step 3] place blue block 3 in yellow bowl 3. ' \
        #     f'[Step 4] done placing blocks in bowls. '

        prompt = \
            f'{prompt}' \
            '[Goal] put the yellow blocks in red bowls. ' \
            '[Initial State] [There is a blue bowl 1, blue bowl 2, blue bowl 3, ' \
            'red bowl 1, red bowl 2, red bowl 3, yellow block 1, ' \
            'yellow block 2, yellow block 3, red block 1. ' \
            f'[Step 1] place yellow block 1 in red bowl 1. ' \
            f'[Step 2] place yellow block 2 in red bowl 2. ' \
            f'[Step 3] place yellow block 3 in red bowl 3. ' \
            f'[Step 4] done placing blocks in bowls. '

        return prompt
