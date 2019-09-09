import matplotlib.pyplot as plt
file = open('Cheetah/2x3-default/console')
generation_default = []
score_default = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation_default.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score_default.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.legend()
plt.show()


file = open('Cheetah/4x3-default/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
file.close()

#plt.figure(figsize=(14,9))
plt.title('Cheetah Runner')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='red', label='4x3-default')
plt.legend()
plt.show()


file = open('Cheetah/2x6-default/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='blue', label='2x6-default')
plt.legend()
plt.show()

######
file = open('Cheetah/2x3-(ruolette_selection)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation, score, color='green', label='2x3-(roulette_selection)')
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(population-500)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation, score, color='gold', label='2x3-(population-500)')
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(population-350)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='pink', label='2x3-(population-350)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(p_mutation=1.0)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='brown', label='2x3-(p_mutation=1.0)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(p_mutation=0.5)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='teal', label='2x3-(p_mutation=0.5)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(mutation-scale=1)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='indigo', label='2x3-(mutation-scale=1)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(mutation_scale=5)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='violet', label='2x3-(mutation_scale=5)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(keep_parents_alive=False)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='orange', label='2x3-(keep_parents_alive=False)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(cutoff=0.6)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='silver', label='2x3-(cutoff=0.6)')
plt.legend()
plt.show()

file = open('Cheetah/2x3-(cutoff=0.4)/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.plot(generation, score, color='olive', label='2x3-(cutoff=0.4)')
plt.legend()
plt.show()

file = open('Cheetah/1x3 - default/result')
generation = []
score = []
for line in file:
    if line[0]=='-':
        gen = line.split(' ')
        generation.append(int(gen[2]))
    if line[:7]=='Total r':
        reward = line.split(':')
        score.append(float(reward[1][:-1]))
plt.plot(generation, score, color='plum', label='1x3 - default')
plt.plot(generation_default, score_default, color='purple', label='2x3-default')
plt.legend()
plt.show()