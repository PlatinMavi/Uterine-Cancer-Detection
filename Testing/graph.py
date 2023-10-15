import matplotlib.pyplot as plt

# Data for test1
test1_combinatual_score = [73.77049180327869, 75.40983606557377, 77.04918032786885]
test1_individual_score = [60.65573770491803, 49.18032786885246, 63.934426229508205, 60.65573770491803,
                          65.57377049180327, 62.295081967213115, 49.18032786885246, 63.934426229508205,
                          62.295081967213115, 63.934426229508205, 60.65573770491803, 62.295081967213115,
                          63.934426229508205, 72.1311475409836, 59.01639344262295, 54.09836065573771,
                          62.295081967213115, 70.49180327868852]

# Data for test2
test2_combinatual_score = [70.49180327868852, 78.68852459016394, 81.9672131147541]
test2_individual_score = [49.18032786885246, 44.26229508196721, 57.377049180327866, 45.90163934426229,
                          52.459016393442624, 63.934426229508205, 42.62295081967213, 59.01639344262295,
                          67.21311475409836, 59.01639344262295, 50.81967213114754, 57.377049180327866,
                          63.934426229508205, 62.295081967213115, 54.09836065573771, 54.09836065573771,
                          59.01639344262295, 68.85245901639344]

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot test1 data
ax1.plot(test1_combinatual_score, marker='o', label='Combinatual Score', color='b')
ax1.set_xticks(range(len(test1_combinatual_score)))
ax1.set_xticklabels(range(1, len(test1_combinatual_score) + 1))
ax1.set_ylabel('Combinatual Score')
ax1.legend(loc='upper right')

ax2.bar(range(len(test1_individual_score)), test1_individual_score, label='Individual Score', color='g')
ax2.set_xticks(range(len(test1_individual_score)+1))
ax2.set_xticklabels(['WBC', 'NEU', 'LYM', 'MONO', 'EOS', 'BASO', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDWSD', 'RDWCV', 'PLT', 'MPV', 'PCT', 'PDW', 'NRBC'])
ax2.set_ylabel('Individual Score')
ax2.legend(loc='upper right')

# Add a title and adjust the layout
plt.suptitle('Test 1')
plt.tight_layout()
plt.show()

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot test2 data
ax1.plot(test2_combinatual_score, marker='o', label='Combinatual Score', color='b')
ax1.set_xticks(range(len(test2_combinatual_score)))
ax1.set_xticklabels(range(1, len(test2_combinatual_score) + 1))
ax1.set_ylabel('Combinatual Score')
ax1.legend(loc='upper right')

ax2.bar(range(len(test2_individual_score)), test2_individual_score, label='Individual Score', color='g')
ax2.set_xticks(range(len(test2_individual_score)+1))
ax2.set_xticklabels(['WBC', 'NEU', 'LYM', 'MONO', 'EOS', 'BASO', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDWSD', 'RDWCV', 'PLT', 'MPV', 'PCT', 'PDW', 'NRBC'])
ax2.set_ylabel('Individual Score')
ax2.legend(loc='upper right')

# Add a title and adjust the layout
plt.suptitle('Test 2')
plt.tight_layout()
plt.show()
