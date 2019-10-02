#!/usr/bin/env python
# coding: utf-8

# In[4]:


from cave.cavefacade import CAVE
import os


# In[5]:


output_dir = "smac_output"
results_dir = os.path.join(output_dir, 'run_1')


# In[6]:


cave = CAVE(folders=[results_dir],
            output_dir="test_cave_smac",
            ta_exec_dir=["."],
            file_format='SMAC3',
           )

