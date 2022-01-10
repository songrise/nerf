import json
import os
# set working directory


line_cnt = 0


def gen_json(t):
    global line_cnt
    print(os.getcwd())
    # read log file
    with open(t+'/m.txt', 'r') as f:
        log = f.read()
        data = {"camera_angle_x": 0.6911112070083618}
        frames = []
        for line in log.split('\n'):

            try:
                record = {"file_path": "./{}/{:04d}_shot".format(
                    t, line_cnt), "rotation": 4.0, "transform_matrix": eval(line)}
                frames.append(record)
            except Exception as e:
                print(e)
            line_cnt += 1
        data["frames"] = frames

        data_json = json.dumps(data)
        with open('transforms_'+t+".json", 'w') as ff:
            ff.write(data_json)


gen_json('train')
gen_json('test')
gen_json('val')
