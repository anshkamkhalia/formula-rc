# @article{CHE2025110694,
#      title = {TwinLiteNet+: An enhanced multi-task segmentation model for autonomous driving},
#      journal = {Computers and Electrical Engineering},
#      volume = {128},
#      pages = {110694},
#      year = {2025},
#      issn = {0045-7906},
#      doi = {https://doi.org/10.1016/j.compeleceng.2025.110694},
#      url = {https://www.sciencedirect.com/science/article/pii/S0045790625006378}
# }

# Copied unmodified from TwinLiteNetPlus:
# https://github.com/chequanghuy/TwinLiteNetPlus
#
# Copyright (c) 2025 Chế Quang Huy
# Licensed under the MIT License. See licenses/TwinLiteNet-LICENSE for full text.


chanel_img = 3
sc_ch_dict = {
    "nano": {  'p': 1,
            'q': 1,
            'chanels' : [4,8, 16, 32, 64],
    },
    
    "small": {  'p': 2,
            'q': 3,
            'chanels' : [8,16, 32, 64, 128],
    },

    "medium": {  'p': 3,
            'q': 5,
            'chanels' : [16,32, 64, 128, 256],
    },

    "large": {  'p': 5,
            'q': 7,
            'chanels' : [32,64, 128, 256, 512],
    }
}