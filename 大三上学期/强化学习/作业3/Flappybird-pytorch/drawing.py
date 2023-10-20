import csv
import matplotlib.pyplot as plt
import numpy as np


csv_reader = csv.reader(open('dqn_reward.csv'))

episode_dqn = []
column_num_dqn = []
for i , line in enumerate(csv_reader):
    if i % 2 != 0:
        continue
    else:
        episode_dqn.append(int(line[0]))
        column_num_dqn.append(int(line[1]))

csv_reader = csv.reader(open('target_dqn_reward_2.csv'))

episode_ddqn = []
column_num_ddqn = []
for i , line in enumerate(csv_reader):
    if i % 2 != 0:
        continue
    else:
        episode_ddqn.append(int(line[0]))
        column_num_ddqn.append(int(line[1]))

plt.figure(dpi=500)
plt.xlabel('Episode')
plt.ylabel('ColumnNum')

# ax = plt.gca()
# x_locator = plt.MultipleLocator(5000)
# y_locator = plt.MultipleLocator(10)
# ax.xaxis.set_major_locator(x_locator)
# ax.yaxis.set_major_locator(y_locator)

# subplot1 = plt.subplot(121)
# subplot1.set_title("DQN")
# subplot1.plot(column_num_dqn , color = 'pink')

# subplot2 = plt.subplot(122)
# subplot2.set_title("Target_DQN")
# subplot2.plot(column_num_ddqn)

plt.plot( episode_ddqn , column_num_ddqn   , label = 'Target_DQN')
plt.plot( episode_dqn , column_num_dqn  ,  color = 'pink' ,label = 'DQN')
plt.legend()
plt.savefig('Result_2.jpg')
# plt.savefig('Result_2_sub.jpg')
plt.show()
