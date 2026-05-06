dataset_info = dict(
    dataset_name='syn_zebras',
    flip_pairs=[
        [0, 3],  # left back paw  <-> right back paw
        [1, 4],  # left back knee <-> right back knee
        [2, 5],  # left back thigh <-> right back thigh
        [6, 9],  # right front paw <-> left front paw
        [7, 10],  # right front knee <-> left front knee
        [8, 11],  # right front thigh <-> left front thigh
        [14, 16],  # right ear tip <-> left ear tip
        [15, 17],  # right ear base <-> left ear base
        [18, 19],  # right eye <-> left eye
    ],
    skeleton=[
        [0, 1],  # left back paw — left back knee
        [1, 2],  # left back knee — left back thigh
        [2, 25],  # left back thigh — back end
        [3, 4],  # right back paw — right back knee
        [4, 5],  # right back knee — right back thigh
        [5, 25],  # right back thigh — back end
        [6, 7],  # right front paw — right front knee
        [7, 8],  # right front knee — right front thigh
        [9, 10],  # left front paw — left front knee
        [10, 11],  # left front knee — left front thigh
        [12, 13],  # tail end — tail base
        [14, 15],  # right ear tip — right ear base
        [16, 17],  # left ear tip — left ear base
        [15, 18],  # right ear base — right eye
        [18, 19],  # right eye — left eye
        [17, 19],  # left ear base — left eye
        [18, 20],  # right eye — nose
        [19, 20],  # left eye — nose
        [18, 23],  # right eye — skull
        [19, 23],  # left eye — skull
        [20, 23],  # nose — skull
        [23, 22],  # skull — neck end
        [22, 21],  # neck end — neck start
        [21, 26],  # neck start — back front
        [26, 8],  # back front — right front thigh
        [26, 11],  # back front — left front thigh
        [26, 24],  # back front — body middle
        [24, 25],  # body middle — back end
        [25, 13],  # back end — tail base
    ],
    pose_kpt_color = [
        [0,255,255],  # 0 left back paw
        [0,255,255],  # 1 left back knee
        [0,255,255],  # 2 left back thigh
        [255,0,255],  # 3 right back paw
        [255,0,255],  # 4 right back knee
        [255,0,255],  # 5 right back thigh
        [0,0,255],    # 6 right front paw
        [0,0,255],    # 7 right front knee
        [0,0,255],    # 8 right front thigh
        [0,255,0],    # 9 left front paw
        [0,255,0],    # 10 left front knee
        [0,255,0],    # 11 left front thigh
        [255,0,0],    # 12 tail end
        [255,0,0],    # 13 tail base
        [255,255,0],  # 14 right ear tip
        [255,255,0],  # 15 right ear base
        [50,128,128], # 16 left ear tip
        [50,128,128], # 17 left ear base
        [255,0,50],   # 18 right eye
        [50,0,255],   # 19 left eye
        [125,50,125], # 20 nose
        [255,0,0],    # 21 neck start
        [255,0,0],    # 22 neck end
        [255,0,0],    # 23 skull
        [255,0,0],    # 24 body middle
        [255,0,0],    # 25 back end
        [255,0,0],    # 26 back front
    ],
    pose_link_color = [
        [0,255,255],  #  0 left back paw - left back knee
        [0,255,255],  #  1 left back knee - left back thigh
        [0,255,255],  #  2 left back thigh - back end
        [255,0,255],  #  3 right back paw - right back knee
        [255,0,255],  #  4 right back knee - right back thigh
        [255,0,255],  #  5 right back thigh - back end
        [255,0,255],  #  6 right front paw - right front knee
        [255,0,255],  #  7 right front knee - right front thigh
        [0,255,255],  #  8 left front paw - left front knee
        [0,255,255],  #  9 left front knee - left front thigh
        [0,0,255],    # 10 tail end - tail base
        [255,0,255],  # 11 right ear tip - right ear base
        [0,255,255],  # 12 left ear tip - left ear base
        [255,0,255],  # 13 right ear base - right eye
        [255,0,255],  # 14 right eye - left eye
        [0,255,255],  # 15 left ear base - left eye
        [255,0,255],  # 16 right eye - nose
        [0,255,255],  # 17 left eye - nose
        [255,0,255],  # 18 right eye - skull
        [0,255,255],  # 19 left eye - skull
        [0,0,255],    # 20 nose - skull
        [0,0,255],    # 21 skull - neck end
        [0,0,255],    # 22 neck end - neck start
        [0,0,255],    # 23 neck start - back front
        [0,0,255],    # 24 back front - right front thigh
        [0,0,255],    # 25 back front - left front thigh
        [0,0,255],    # 26 back front - body middle
        [0,0,255],    # 27 body middle - back end
        [0,0,255],    # 28 back end - tail base
    ],
    keypoint_info={
        0:
        dict(
            name='left back paw',
            id=0,
            color=[0,255,255],
            type='lower',
            swap='right back paw'),
        1:
        dict(
            name='left back knee',
            id=1,
            color=[0,255,255],
            type='lower',
            swap='right back knee'),
        2:
        dict(name='left back thigh',
             id=2,
             color=[0,255,255],
             type='lower',
             swap='right back thigh'),
        3:
        dict(name='right back paw',
             id=3,
             color=[255,0,255],
             type='lower',
             swap='left back paw'),
        4:
        dict(
            name='right back knee',
            id=4,
            color=[255,0,255],
            type='lower',
            swap='left back knee'),
        5:
        dict(
            name='right back thigh',
            id=5,
            color=[255,0,255],
            type='lower',
            swap='left back thigh'),
        6:
        dict(
            name='right front paw',
            id=6,
            color=[0,0,255],
            type='upper',
            swap='left front paw'),
        7:
        dict(
            name='right front knee',
            id=7,
            color=[0, 0,255],
            type='upper',
            swap='left front knee'),
        8:
        dict(
            name='right front thigh',
            id=8,
            color=[0, 0, 255],
            type='upper',
            swap='left front thigh'),
        9:
        dict(
            name='left front paw',
            id=9,
            color=[0,255,0],
            type='upper',
            swap='right front paw'),
        10:
        dict(
            name='left front knee',
            id=10,
            color=[0,255,0],
            type='upper',
            swap='right front knee'),
        11:
        dict(
            name='left front thigh',
            id=11,
            color=[0,255,0],
            type='upper',
            swap='right front thigh'),
        12:
        dict(
            name='tail end',
            id=12,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        13:
        dict(
            name='tail base',
            id=13,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        14:
        dict(
            name='right ear tip',
            id=14,
            color=[255,255,0],
            type='upper',
            swap='left ear tip'),
        15:
        dict(
            name='right ear base',
            id=15,
            color=[255,255,0],
            type='upper',
            swap='left ear base'),
        16:
        dict(
            name='left ear tip',
            id=16,
            color=[50,128,128],
            type='upper',
            swap='right ear tip'),
        17:
        dict(
            name='left ear base',
            id=17,
            color=[50,128,128],
            type='upper',
            swap='right ear base'),
        18:
        dict(
            name='right eye',
            id=18,
            color=[255,0,50],
            type='upper',
            swap='left eye'),
        19:
        dict(
            name='left eye',
            id=19,
            color=[50,0,255],
            type='upper',
            swap='right eye'),
        20:
        dict(
            name='nose',
            id=20,
            color=[125,50, 125],
            type='upper',
            swap=''),
        21:
        dict(
            name='neck start',
            id=21,
            color=[255,0, 0],
            type='upper',
            swap=''),
        22:
        dict(
            name='neck end',
            id=22,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        23:
        dict(
            name='skull',
            id=23,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        24:
        dict(
            name='body middle',
            id=24,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        25:
        dict(
            name='back end',
            id=25,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        26:
        dict(
            name='back front',
            id=26,
            color=[255, 0, 0],
            type='upper',
            swap='')
    },
    skeleton_info={
        0: dict(link=('left back paw', 'left back knee'), id=0, color=[0, 255, 255]),
        1: dict(link=('left back knee', 'left back thigh'), id=1, color=[0, 255, 255]),
        2: dict(link=('left back thigh', 'back end'), id=2, color=[0, 255, 255]),
        3: dict(link=('right back paw', 'right back knee'), id=3, color=[255, 0, 255]),
        4: dict(link=('right back knee', 'right back thigh'), id=4, color=[255, 0, 255]),
        5: dict(link=('right back thigh', 'back end'), id=5, color=[255, 0, 255]),
        6: dict(link=('right front paw', 'right front knee'), id=6, color=[255, 0, 255]),
        7: dict(link=('right front knee', 'right front thigh'), id=7, color=[255, 0, 255]),
        8: dict(link=('left front paw', 'left front knee'), id=8, color=[0, 255, 255]),
        9: dict(link=('left front knee', 'left front thigh'), id=9, color=[0, 255, 255]),
        10: dict(link=('tail end', 'tail base'), id=10, color=[0, 0, 255]),
        11: dict(link=('right ear tip', 'right ear base'), id=11, color=[255, 0, 255]),
        12: dict(link=('left ear tip', 'left ear base'), id=12, color=[0, 255, 255]),
        13: dict(link=('right ear base', 'right eye'), id=13, color=[255, 0, 255]),
        14: dict(link=('right eye', 'left eye'), id=14, color=[255, 0, 255]),
        15: dict(link=('left ear base', 'left eye'), id=15, color=[0, 255, 255]),
        16: dict(link=('right eye', 'nose'), id=16, color=[255, 0, 255]),
        17: dict(link=('left eye', 'nose'), id=17, color=[0, 255, 255]),
        18: dict(link=('right eye', 'skull'), id=18, color=[255, 0, 255]),
        19: dict(link=('left eye', 'skull'), id=19, color=[0, 255, 255]),
        20: dict(link=('nose', 'skull'), id=20, color=[0, 0, 255]),
        21: dict(link=('skull', 'neck end'), id=21, color=[0, 0, 255]),
        22: dict(link=('neck end', 'neck start'), id=22, color=[0, 0, 255]),
        23: dict(link=('neck start', 'back front'), id=23, color=[0, 0, 255]),
        24: dict(link=('back front', 'right front thigh'), id=24, color=[0, 0, 255]),
        25: dict(link=('back front', 'left front thigh'), id=25, color=[0, 0, 255]),
        26: dict(link=('back front', 'body middle'), id=26, color=[0, 0, 255]),
        27: dict(link=('body middle', 'back end'), id=27, color=[0, 0, 255]),
        28: dict(link=('back end', 'tail base'), id=28, color=[0, 0, 255]),
    },
    # joint_weights=[1.0, 1.0, 1.0, # lb paw, lb knee, lb thigh
    #                1.0, 1.0, 1.0, # rb paw, rb knee, rb thigh
    #                1.0, 1.0, 1.0, # rf paw, rf knee, rf thigh
    #                1.0, 1.0, 1.0, # lf paw, lf knee, lf thigh
    #                1.0, 1.0, # tail end, tail base
    #                1.0, 1.0, # right ear tip, right ear base
    #                1.0, 1.0, # left ear tip, left ear base
    #                1.0, 1.0, 1.0, # right eye, left eye, nose
    #                1.0, 1.0, 1.0, # neck start, neck end, skull
    #                1.0, 1.0, 1.0], # body middle, back end, back front
    joint_weights=[1.2, 1.2, 1.0, # lb paw, lb knee, lb thigh
                   1.5, 1.5, 1.0, # rb paw, rb knee, rb thigh
                   1.5, 1.5, 1.0, # rf paw, rf knee, rf thigh
                   1.2, 1.2, 1.0, # lf paw, lf knee, lf thigh
                   1.5, 0.9, # tail end, tail base
                   0.9, 0.7, # right ear tip, right ear base
                   0.9, 0.7, # left ear tip, left ear base
                   0.5, 0.5, 0.9, # right eye, left eye, nose
                   0.6, 0.6, 0.6, # neck start, neck end, skull
                   0.6, 0.6, 0.6], # body middle, back end, back front
     sigmas=[
        0.089, 0.087, 0.107, # lb paw, lb knee, lb thigh
        0.089, 0.087, 0.107, # rb paw, rb knee, rb thigh
        0.062, 0.072, 0.079, # rf paw, rf knee, rf thigh
        0.062, 0.072, 0.79, # lf paw, lf knee, lf thigh
        0.1, 0.035, # tail end, tail base
        0.05, 0.025, # right ear tip, right ear base
        0.05, 0.025, # left ear tip, left ear base
        0.025, 0.025, 0.026, # right eye, left eye, nose
        0.035, 0.035, 0.035, # neck start, neck end, skull
        0.05, 0.035, 0.035, # body middle, back end, back front
     ],

    #sigmas=[
    #    0.09, 0.09, 0.1, # lb paw, lb knee, lb thigh
    #    0.09, 0.09, 0.1, # rb paw, rb knee, rb thigh
    #    0.09, 0.09, 0.1, # rf paw, rf knee, rf thigh
    #    0.09, 0.09, 0.1, # lf paw, lf knee, lf thigh
    #    0.1, 0.035, # tail end, tail base
    #    0.05, 0.025, # right ear tip, right ear base
    #    0.05, 0.025, # left ear tip, left ear base
    #    0.025, 0.025, 0.05, # right eye, left eye, nose
    #    0.035, 0.035, 0.035, # neck start, neck end, skull
    #    0.05, 0.025, 0.025, # body middle, back end, back front
    #],
)
