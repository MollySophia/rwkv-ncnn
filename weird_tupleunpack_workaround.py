import os

os.system("mv ./output/model.ncnn.param ./output/model.ncnn.param.old")
old_file = open("./output/model.ncnn.param.old", "r")
new_file = open("./output/model.ncnn.param", "w")

data_read = old_file.readline()
while len(data_read) != 0:
    if "rwkv.rwkv_v4neo.RWKV_Time_Mixing" in data_read:
        next_line = old_file.readline()

        if "prim::TupleUnpack" in next_line:
            data_split = data_read.split()
            next_line_split = next_line.split()

            # output count
            data_split[3] = next_line_split[3]
            del data_split[-1]
            for i in range(int(data_split[3])):
                data_split.append(next_line_split[i - int(data_split[3])])

            new_file.writelines([' '.join(data_split[0:2]) + "           " + ' '.join(data_split[2:]) + '\n'])

        else:
            new_file.write(data_read)
            new_file.write(next_line)
    else:
        new_file.write(data_read)

    data_read = old_file.readline()

new_file.flush()
old_file.close()
new_file.close()
