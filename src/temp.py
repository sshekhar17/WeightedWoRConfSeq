import os 

module_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(module_dir)
fig_dir = parent_dir + '/data'
print(fig_dir)