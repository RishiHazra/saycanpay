class PromptRavensTowersOfHanoiSeq:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        # prompt = \
        #     'Task: move the brown disk in rod 3. ' \
        #     'Initial State: red disk on top of brown disk. ' \
        #     'brown disk on top of blue disk. blue disk rod 1. ' \
        #     'The locations for moving are lighter brown side, middle of the stand, ' \
        #     'darker brown side. ' \
        #     f'Step 1:put red disk in middle of the stand. ' \
        #     f'Step 2:put brown disk in darker brown side. ' \
        #     f'Step 3:done putting disks in stands. '
        # prompt = '[Game] Tower of Hanoi. '

        prompt = \
            '[Goal] move the gray disk in rod 2. ' \
            '[Initial State] red disk on top of gray disk. ' \
            'gray disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3. ' \
            '[Step 1] put red disk in rod 3. ' \
            '[Step 2] put gray disk in rod 2. ' \
            '[Step 3] done putting disks in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the yellow disk in rod 3. ' \
            '[Initial State] red disk on top of green disk. ' \
            'green disk on top of yellow disk. yellow disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3.  ' \
            '[Step 1] put red disk in rod 3. ' \
            '[Step 2] put green disk in rod 2. ' \
            '[Step 3] put red disk in rod 2. ' \
            '[Step 4] put yellow disk in rod 3. ' \
            '[Step 5] done putting disks in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the blue disk in rod 2. ' \
            '[Initial State] blue disk on top of cyan disk. cyan disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3. ' \
            '[Step 1] put blue disk in rod 2. ' \
            '[Step 2] done putting disks in rods. '

        # prompt = \
        #     f'{prompt}' \
        #     '[Goal] move the blue disk to the middle of the stand. ' \
        #     '[Initial State] cyan disk on top of brown disk. ' \
        #     'brown disk on top of blue disk. blue disk in rod 1. ' \
        #     'TThe disks can be moved in rod 1, rod 2, rod 3. ' \
        #     '[Step 1] put cyan disk in middle of the stand. ' \
        #     '[Step 2] put brown disk in darker brown side. ' \
        #     '[Step 3] put cyan disk in darker brown side. ' \
        #     '[Step 4] put blue disk in middle of the stand. ' \
        #     '[Step 5] done putting disks in stands. '

        # prompt = \
        #     f'{prompt}' \
        #     'Initial State: brown disk on top of green disk. ' \
        #     'green disk on top of gray disk. gray disk in rod 1. ' \
        #     'The locations for moving are lighter brown side, middle of the stand, ' \
        #     'darker brown side. ' \
        #     'Task: move the gray disk to the middle of the stand. ' \
        #     f'Step 1:put brown disk in middle of the stand. ' \
        #     f'Step 2:put green disk in darker brown side. ' \
        #     f'Step 3:put brown disk in darker brown side. ' \
        #     f'Step 4:put gray disk in middle of the stand. ' \
        #     f'Step 5:done putting disks in stands. '

        return prompt
