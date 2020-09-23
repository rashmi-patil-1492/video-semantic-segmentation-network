# Convert video frame set to 3 coloumn file
import os

frames_size = 30
folder_type = 'test'
folder_names_train = ['01TP_30_1860', '06R0_930_3930', '16E5_390_2400', '16E5_4350_7920', '16E5_8190_8640']
folder_names_val = ['16E5_7959_8159']
folder_names_test = [ '01TP_1890_3720', '05VD_0_5100']
folder_names = folder_names_test
from pathlib import Path
home = str(Path.home())
base_dir = home+'/data/video-segmentation/camvid_fps_30/'

def strip_file_path():
    # return os.getcwd().replace('tools', '')
    return base_dir


def get_file_list(raw_file_folder, ann_file_folder, prefix, file_num_start, file_num_end):

    frame_start = file_num_start
    iter_start = file_num_start - frames_size + 1 if file_num_start - frames_size >= 0 else 0
    iter_end = file_num_end
    filelist = []
    for i in range(iter_start, iter_end):

        if frame_start > file_num_end:
            frame_start = file_num_end

        raw_file_path = raw_file_folder + '/' + prefix + str(i).zfill(6) + '.png'
        last_frame_file_path = raw_file_folder + '/' + prefix + str(frame_start).zfill(6) + '.png'
        ann_file_path = ann_file_folder + '/' + prefix + str(frame_start).zfill(6) + '.png'
        # special case:
        if prefix == '16E5_' and file_num_start == 390 and file_num_end == 2400 and frame_start == 900:
            last_frame_file_path = raw_file_folder + '/' + prefix + str(frame_start + 1).zfill(6) + '.png'
            ann_file_path = ann_file_folder + '/' + prefix + str(frame_start + 1).zfill(6) + '.png'

        if not os.path.isfile(raw_file_path):
            print('missing files', raw_file_path)
            break
        if not os.path.isfile(ann_file_path):
            print('missing files', ann_file_path)
            break
        filelist.append(
            raw_file_path.replace(strip_file_path(), '') + ' ' +
            last_frame_file_path.replace(strip_file_path(), '') +
            ' ' + ann_file_path.replace(strip_file_path(), '') + '\n'
        )
        if i == frame_start:
            frame_start = frame_start + frames_size
    return filelist


file_list = []
for folder in folder_names:
    print('processing folder ', folder)
    splits = folder.split('_')
    file_list += get_file_list(
        raw_file_folder=base_dir + folder_type + '/' + folder,
        ann_file_folder=base_dir + folder_type + 'annot/' + folder,
        prefix=splits[0] + '_',
        file_num_start=int(splits[1]),
        file_num_end=int(splits[2])
    )

# Write to a file
print(len(file_list))
with open(folder_type + '_file_list.txt', 'w') as out:
    for line in file_list:
        out.write(line)
out.close()

print(' done !!!')






