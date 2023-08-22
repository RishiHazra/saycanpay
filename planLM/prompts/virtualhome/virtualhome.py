class PromptVirtualHome:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):

        prompt = \
            '[Goal] write an email. ' \
            '[Step 1] find keyboard. ' \
            '[Step 2] grab keyboard. ' \
            '[Step 3] find computer. ' \
            '[Step 4] switchon computer. ' \
            '[Step 5] done task. '

        prompt = \
            f'{prompt}' \
            '[Goal] wash dishes with dishwasher. ' \
            '[Step 1] walk sink. ' \
            '[Step 2] find plate. ' \
            '[Step 3] grab plate. ' \
            '[Step 4] find dishwasher. ' \
            '[Step 5] open dishwasher. ' \
            '[Step 6] putback plate. ' \
            '[Step 7] close dishwasher. ' \
            '[Step 8] switchon dishwasher. ' \
            '[Step 9] done task. '

        prompt = \
            f'{prompt}' \
            '[Goal] read book. ' \
            '[Step 1] walk novel. ' \
            '[Step 2] find novel. ' \
            '[Step 3] grab novel. ' \
            '[Step 4] find sofa. ' \
            '[Step 5] sit sofa. ' \
            '[Step 6] read novel. ' \
            '[Step 7] done task. '

        # prompt = \
        #     f'{prompt}' \
        #     '[Goal] listen to music. ' \
        #     '[Step 1] walk stereo. ' \
        #     '[Step 2] switchon stereo. ' \
        #     '[Step 3] done task. '

        # prompt = \
        #     f'{prompt}' \
        #     '[Goal] go to toilet. ' \
        #     '[Step 1] walk bathroom. ' \
        #     '[Step 2] walk toilet. ' \
        #     '[Step 3] find toilet. ' \
        #     '[Step 4] done task. '

        return prompt
