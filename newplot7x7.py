import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import trace as d1t
import math

# Read File
def read_bin(filename: str):
    trc1 = d1t.parser(filename).parse()
    d1_touch_tracks = trc1['d1_touch_tracks']
    df = pd.DataFrame(d1_touch_tracks['bar_tracks']['bar_num'][:,0], columns=['bar_num'])
    df['pos0'] = pd.DataFrame(d1_touch_tracks['bar_tracks']['pos0'][:,0] / 16)
    df['pos1'] = pd.DataFrame(d1_touch_tracks['bar_tracks']['pos1'][:,0] / 16)
    df['pos2'] = pd.DataFrame(d1_touch_tracks['bar_tracks']['pos2'][:,0])
    df['force'] = pd.DataFrame(d1_touch_tracks['bar_tracks']['force_lvl'][:,0], dtype=int)
    return df

# Grabbing Data
def read_pos_log(filename: str):
    headers = ['Label' , 'Date', 'Time']
    pos_df = pd.read_csv(filename, sep=' ', header=None, names=headers)
    list_pos, Y_pos = [], ""
    for i in range(len(pos_df)):
        if 'MOVE_Y' in pos_df.iloc[i,0]:
            Y_pos = str(pos_df.iloc[i, 0])[6:]
        if str(pos_df.iloc[i, 0])[:6] == 'MOVE_X':
            X_pos = str(pos_df.iloc[i, 0])[6:]
            pos_start = datetime.datetime.strptime(pos_df.iloc[i+1, 1] + ' ' + pos_df.iloc[i+1, 2],
                                                   '%Y-%m-%d %H:%M:%S.%f')
            pos_end = datetime.datetime.strptime(pos_df.iloc[i+2, 1] + ' ' + pos_df.iloc[i+2, 2],
                                                 '%Y-%m-%d %H:%M:%S.%f')
            list_pos.append([int(X_pos), int(float(Y_pos)), pos_start, pos_end, \
                             (pos_end - pos_start).seconds * 1000000 + (pos_end - pos_start).microseconds])
    data_frame = pd.DataFrame(list_pos , columns=['X', 'Y', 'Start', 'End', 'Dur'])
    return data_frame

# Initalizing
xpoint2, ypoint2, xpoint1, ypoint1, distance_final, force_lvl = [], [], [], [], [], []
id, force, pos0, pos1, df = [], [], [], [], []

fn = '#2843_D5_W100g_s5_10hz_20210623_170143'
if len(sys.argv) > 1:
    fn = str(sys.argv[1])[:-4]
fn_bin = fn + '.bin'
fn_robot = fn + '_robot.txt'
df_bin = read_bin(fn_bin)
df_robot = read_pos_log(fn_robot)

bin_start = datetime.datetime.strptime(fn[-15:], '%Y%m%d_%H%M%S')

T_FRAME = 0.0100775384615385
T_FRAME = 0.01 #775384615385
FRAME_RATE = 10
FACT_RATIO =  1 #  1.63545  for 100z
time_ratio = (100 * T_FRAME) / FRAME_RATE
robot_pos = 0

# DF table
for i in range(len(df_bin)):
    ctime = bin_start + datetime.timedelta(seconds=(i * time_ratio * FACT_RATIO + 0.5))
    Z_down = df_robot.Start[robot_pos]
    Z_up = df_robot.End[robot_pos]
    if Z_down <= ctime < Z_up:
        pos0.append(df_bin.pos0[i])
        pos1.append(df_bin.pos1[i])
        force.append(df_bin.force[i])
        id.append(i)
    if ctime >= Z_up:
        df.append([pos0, pos1, force, id, df_robot.X[robot_pos], df_robot.Y[robot_pos],
                    pos0[-2],pos1[-2],force[-2]])
        pos0, pos1, force, id = [], [], [], []
        robot_pos += 1
        if len(df) >= len(df_robot):
            break

for each in df:
    force_lvl.append(each[-1])

#Plot Position and Force Level
fig, ax = plt.subplots(3)
fig.suptitle(fn.split('\\')[-1]  + '- Pos and Force level')
ax[0].set_title('Pos0')
ax[0].plot(df_bin.pos0, linewidth=1.5, label='Pos0')
ax[0].set_xlabel('Frame Number', fontsize=8)
ax[0].legend()
ax[1].set_title('Pos1')
ax[1].plot(df_bin.pos1,linewidth=1.5, label='Pos1', color='olive')
ax[1].set_xlabel('Frame Number', fontsize=8)
ax[1].legend()
ax[2].set_title('Force Level')
ax[2].plot(df_bin.force, linewidth=1.5, label='Force', color='g')
ax[2].set_xlabel('Frame Number',fontsize=8)
ax[2].legend(loc=1)
df_x, df_y = [], []
for each in df:
    df_x.append(int(each[0][-2]))
    df_y.append(int(each[1][-2]))
    ax[0].plot(each[3][-2:], each[0][-2:], linewidth=3, color='red')
    ax[1].plot(each[3][-2:], each[1][-2:], linewidth=3, color='red')
    ax[2].plot(each[3][-2:], each[2][-2:], linewidth=3, color='red')
    #xx.plot(each[3][-2:], each[2][-2:], linewidth=4, color='red')
fig.set_size_inches(9.0, 7.0)
plt.tight_layout()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(fn + '_pos_force.png')


#HeatMap Force Level
fig2, bx = plt.subplots(1)
cmap_color = 'Greens_r'
force_lvl = np.reshape(force_lvl, (len(np.unique(df_robot.Y)), len(np.unique(df_robot.X))))
x_labels = np.unique(df_robot.X)
y_labels = np.unique(df_robot.Y)
if len(x_labels) > 10:
    fig2.set_size_inches(9.8,8.0)
    sns.set(font_scale=.5)
bx.set_title(fn.split('\\')[-1]  + '\nForce Level  Heat Map')
bx = sns.heatmap(force_lvl,
                 annot=True,
                 fmt=".0f",
                 cmap=cmap_color,
                 linewidths=0.5)
bx.set_xticklabels(x_labels)
bx.set_yticklabels(y_labels)
fig2.savefig(fn + '_heatmap.png')

#Grid Map Robot Location and Dpad Location
fig3, cx = plt.subplots(1)
cx.plot(df_robot.X.astype(int),
        df_robot.Y.astype(int),
        'ro',
        markersize=10,
        mfc='none',
        label='robot_ref')
cx.set_title(fn.split('\\')[-1] + '\nRobot loc Vs. Dpad loc')
cx.plot(df_x,
        df_y,
        'go',
        markersize=10,
        label='Pos0:Pos1')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylim(bottom=-18, top=18)
plt.xlim(-18,18)
plt.xlabel('X (mm)', fontsize=8)
plt.ylabel('Y (mm)', fontsize=8)
plt.tick_params(labeltop=True, labelright=True)
plt.grid(b=True,
        which='major',
        color='#666666',
        linestyle='-')
plt.tight_layout()
fig3.savefig(fn + '_dpad_robot.png')

# Heatmap Distance
for i in range(0, len(df)):
    xpoint2.append(df[i][6])
    ypoint2.append(df[i][7])
    xpoint1.append(df[i][4])
    ypoint1.append(df[i][5])
distancev1 = lambda x1,y1,x2,y2: math.sqrt( ((x1[i]-x2[i])**2) + ((y1[i]-y2[i])**2))
for i in range(len(xpoint1)):
    distance_final.append( distancev1(xpoint1,ypoint1,xpoint2,ypoint2) )
distance_map = np.reshape(distance_final,(7,7))
fig, cx = plt.subplots(1)
cx.set_title("Error Distance")
cx = sns.heatmap(distance_map, annot=True,
                        fmt=".0f",
                        cmap='Oranges_r',
                        vmax=50,
                        vmin=-50,
                        linewidths=0.5,
                        )

plt.pause(1200)